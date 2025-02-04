import logging
import random
import math
import warnings
import PIL
from pathlib import Path
from typing import Union
import numpy as np
import re
import cv2
import lmdb
import six
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

from semimtr.utils.transforms import CVColorJitter, CVDeterioration, CVGeometry
from semimtr.utils.utils import CharsetMapper, onehot


class ImageDataset(Dataset):
    '''
    `ImageDataset` read data from LMDB database, resize image and truncate label.
    Support resize type:
        - `consistent`: resize images to same shape
        - `varied`: resize images to same height, keep ratio and round width to multiple of 8
        - `padded`: resize images to same height, pad images by background to have same width
    '''
    def __init__(self,
                 path: Union[Path, str],
                 is_training: bool = True,
                 img_h: int = 32,
                 img_w: int = 100,
                 resize_type: str = 'consistent',
                 max_length: int = 25, # Fix this
                 space_as_token: bool = False,
                 check_length: bool = True,
                 filter_single_punctuation: bool = False,
                 case_sensitive: bool = False,
                 charset_path: str = 'data/charset_36.txt',
                 convert_mode: str = 'RGB',
                 data_aug: bool = True,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 data_portion: float = 1.0,
                 **kwargs):
        self.path, self.name = Path(path), Path(path).name
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode, self.check_length = convert_mode, check_length
        # By default, image min width = image height
        self.img_h, self.img_min_w, self.img_max_w = img_h, img_h, img_w
        # Resize type will affect how image is standardize
        assert resize_type in ['consistent', 'varied', 'padded'], \
            f'{resize_type} is not supported'
        self.resize_type = resize_type
        self.max_length, self.one_hot_y = max_length, one_hot_y
        self.case_sensitive, self.is_training = case_sensitive, is_training
        # Filter punctuation
        self.filter_single_punctuation = filter_single_punctuation
        # Data augmentation for training and multiscale if needed
        self.data_aug, self.multiscales = data_aug, multiscales
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1, space_as_token=space_as_token)
        self.charset_string = ''.join([*self.charset.char_to_label])
        # Escaping the hyphen for later use in regex
        self.charset_string = re.sub('-', r'\-', self.charset_string)  
        self.c = self.charset.num_classes

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            dataset_length = int(txn.get('num-samples'.encode()))
        self.use_portion = self.is_training and not data_portion == 1.0
        if not self.use_portion:
            self.length = dataset_length
        else:
            self.length = int(data_portion * dataset_length)
            self.optional_ind = np.random.permutation(dataset_length)[:self.length]

        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return self.length

    def _next_image(self):
        if not self.is_training:
            return
        next_index = random.randint(0, len(self) - 1)
        if self.use_portion:
            next_index = self.optional_ind[next_index]
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def standardize_width(self, img):
        # As expected, Dataset will only use this to process images
        # in np.array type, which means this image is just read and converted to np,
        # and currently has no other applied operation. Any other type needs to be rechecked.
        if len(img.shape)==3:
            current_h, current_w, _ = img.shape
        else:
            raise Exception('Something went wrong here!!!')
        round_to = 8
        new_w = math.ceil(self.img_h * current_w/current_h / round_to) * round_to
        new_w = np.clip(new_w, self.img_min_w, self.img_max_w)
        return new_w

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        # Review this later
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_max_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_max_w)
                trg_w = self.img_max_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_max_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_max_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.resize_type == 'consistent':
            if self.multiscales:
                return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
            else:
                return cv2.resize(img, (self.img_max_w, self.img_h))
        elif self.resize_type == 'varied':
            new_w = self.standardize_width(img)
            return cv2.resize(img, (new_w, self.img_h))
        elif self.resize_type == 'padded':
            color = np.mean([img[0,0], img[-1,-1], img[0,-1], img[-1,0]], axis=0)

            new_w = self.standardize_width(img)
            new_im = cv2.resize(img, (new_w, self.img_h))
            # Pad image with a solid color
            padding = np.tile(color,(self.img_h, self.img_max_w-new_w, 1))
            return np.concatenate([new_im, padding],axis=1).astype(np.uint8)
        else:
            raise NotImplementedError()

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            exception_flag = False
            try:
                raw_label = str(txn.get(label_key.encode()), 'utf-8')  # label
                if not self.case_sensitive: raw_label = raw_label.lower()
                label = re.sub(f'[^{self.charset_string}]', '', raw_label)
                # label = re.sub('[^0-9a-zA-Z]+', '', raw_label)

                # Remove image with too long label or missing-label
                len_issue = 0 < self.max_length < len(label) or len(label) <= 0

                # Remove label has length=1 and is a punctuation 
                single_punctuation = len(label) == 1 and label in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~° '

                if (self.check_length and len_issue) or (self.filter_single_punctuation and single_punctuation):
                    return self._next_image()
                # Truncate label by max_length. Increase max_length to train with text line.
                label = label[:self.max_length]

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
            except:
                import traceback
                traceback.print_exc()
                exception_flag = True
                if "label" in locals():
                    logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                else:
                    logging.info(f'Corrupted image is found: {self.name}, {idx}')
                return self._next_image()
            if exception_flag or not self._check_image(image):
                return self._next_image() if self.is_training else None
            return {'image': image, 'label': label, 'idx': idx}

    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.totensor(self.resize(np.array(image)))
        return image

    def _process_test(self, image):
        return self.totensor(self.resize(np.array(image)))

    def __getitem__(self, idx):
        if self.use_portion:
            idx = self.optional_ind[idx]
        datum = self.get(idx)
        if datum is None:
            return
        image, text, idx_new = datum['image'], datum['label'], datum['idx']

        if self.is_training:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        y = self._label_postprocessing(text)
        return image, y

    def _label_postprocessing(self, text):
        length = torch.tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label = torch.tensor(label).to(dtype=torch.long)
        if self.one_hot_y: label = onehot(label, self.charset.num_classes)
        return {'label': label, 'length': length}


class TextDataset(Dataset):
    def __init__(self,
                 path: Union[Path, str],
                 delimiter: str = '\t',
                 max_length: int = 25,
                 charset_path: str = 'data/charset_36.txt',
                 space_as_token: bool = False,
                 case_sensitive=False,
                 one_hot_x=True,
                 one_hot_y=True,
                 is_training=True,
                 smooth_label=False,
                 smooth_factor=0.2,
                 use_sm=False,
                 **kwargs):
        self.path = Path(path)
        # Case sensitive & spelling mutation
        self.case_sensitive, self.use_sm = case_sensitive, use_sm
        # Convert hard label to smooth label
        self.smooth_factor, self.smooth_label = smooth_factor, smooth_label
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1,space_as_token=space_as_token)
        # convert the charset to string for regex filtering
        self.charset_string = ''.join([*self.charset.char_to_label])
        # escaping the hyphen for later use in regex
        self.charset_string = re.sub('-', r'\-', self.charset_string)  
        self.one_hot_x, self.one_hot_y, self.is_training = one_hot_x, one_hot_y, is_training
        # Define spelling mutation: Apply random insert/delete/replace characters
        if self.is_training and self.use_sm: self.sm = SpellingMutation(charset=self.charset)

        dtype = {'inp': str, 'gt': str}
        self.df = pd.read_csv(self.path, dtype=dtype, delimiter=delimiter, na_filter=False,quoting=3)
        self.inp_col, self.gt_col = 0, 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_x = self.df.iloc[idx, self.inp_col]
        if not self.case_sensitive: text_x = text_x.lower()
        text_x = re.sub(f'[^{self.charset_string}]', '', text_x)
        if self.is_training and self.use_sm: text_x = self.sm(text_x)

        length_x = torch.tensor(len(text_x) + 1).to(dtype=torch.long)  # one for end token
        label_x = self.charset.get_labels(text_x, case_sensitive=self.case_sensitive)
        label_x = torch.tensor(label_x)
        if self.one_hot_x:
            label_x = onehot(label_x, self.charset.num_classes)
            if self.is_training and self.smooth_label: # Convert one hot to smooth label
                label_x = torch.stack([self.prob_smooth_label(l) for l in label_x])
        x = {'label': label_x, 'length': length_x}

        text_y = self.df.iloc[idx, self.gt_col]
        if not self.case_sensitive: text_y = text_y.lower()
        text_y = re.sub(f'[^{self.charset_string}]', '', text_y)
        length_y = torch.tensor(len(text_y) + 1).to(dtype=torch.long)  # one for end token
        label_y = self.charset.get_labels(text_y, case_sensitive=self.case_sensitive)
        label_y = torch.tensor(label_y)
        if self.one_hot_y: label_y = onehot(label_y, self.charset.num_classes)
        y = {'label': label_y, 'length': length_y}
        return x, y

    def prob_smooth_label(self, one_hot):
        one_hot = one_hot.float()
        delta = torch.rand([]) * self.smooth_factor
        num_classes = len(one_hot)
        noise = torch.rand(num_classes)
        noise = noise / noise.sum() * delta
        # noise = delta * dist: this meant delta is divide to a list that sum=delta
        # one_hot * (1 - delta): this meant label =[0,0,...,1-delta,..,0,0]
        # smoothed_label = label + noise
        one_hot = one_hot * (1 - delta) + noise
        return one_hot


class SpellingMutation(object):
    def __init__(self, pn0=0.7, pn1=0.85, pn2=0.95, pt0=0.7, pt1=0.85, charset=None):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        self.charset = charset
        logging.info(f'the probs: pn0={self.pn0}, pn1={self.pn1} ' +
                     f'pn2={self.pn2}, pt0={self.pt0}, pt1={self.pt1}')

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.charset.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return (char not in self.charset.digits) and (char not in self.charset.alphabets)

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0:
            num_to_modify = 0
        elif prob < self.pn1:
            num_to_modify = 1
        elif prob < self.pn2:
            num_to_modify = 2
        else:
            num_to_modify = 3

        if length <= 1:
            num_to_modify = 0
        elif length >= 2 and length <= 4:
            num_to_modify = min(num_to_modify, 1)
        else:
            num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index:
                chars.append(t)
            elif self.is_unk_char(t):
                chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0:  # replace
                    chars.append(random.choice(self.charset.alphabets))
                elif prob < self.pt1:  # insert
                    chars.append(random.choice(self.charset.alphabets))
                    chars.append(t)
                else:  # delete
                    continue
        new_text = ''.join(chars[: self.charset.max_length - 1])
        return new_text if len(new_text) >= 1 else text


def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)