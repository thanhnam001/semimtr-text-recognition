import math
import torch
import cv2
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
from PIL import Image
from semimtr.dataset.dataset import ImageDataset
from semimtr.dataset.weighted_sampler import WeightedDatasetRandomSampler
from torch.utils.data.sampler import Sampler


class ImageDatasetLine(ImageDataset):
    def __init__(self, 
                 img_min_w: int = 32,
                 img_max_w: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.img_min_w = img_min_w
        self.img_max_w = img_max_w

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

    def resize(self, img):
        new_w = self.standardize_width(img)
        return cv2.resize(img, (new_w, self.img_h))
    
class ImageDatasetLineV2(ImageDataset):
    def __init__(self, 
                 img_min_w: int = 32,
                 img_max_w: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.img_min_w = img_min_w
        self.img_max_w = img_max_w

    def standardize_width(self, img):
        if len(img.shape)==3:
            current_h, current_w, _ = img.shape
        else:
            raise Exception('Something went wrong here!!!')
        round_to = 8
        new_w = math.ceil(self.img_h * current_w/current_h / round_to) * round_to
        new_w = np.clip(new_w, self.img_min_w, self.img_max_w)
        return new_w

    def resize(self, img):
        color = np.mean([img[0,0], img[-1,-1], img[0,-1], img[-1,0]], axis=0)

        new_w = self.standardize_width(img)
        new_im = cv2.resize(img, (new_w, self.img_h))
        # Pad image with a solid color
        padding = np.tile(color,(self.img_h, self.img_max_w-new_w, 1))
        return np.concatenate([new_im, padding],axis=1).astype(np.uint8)

class ClusterRandomSampler(Sampler[int]):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.build_cluster_indices()

    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)

        pbar = tqdm(range(len(self.data_source)), 
                desc='{} build cluster'.format(1), 
                ncols = 100, position=0, leave=True) 
        # The key idea here is group sample's index by its width
        for i in pbar:
            image, _ = self.data_source[i]
            _, _, _, bucket = image.shape # (B, C, H, W)
            self.cluster_indices[bucket].append(i)

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        # For each cluster, shuffle its index
        for cluster, cluster_indices in self.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)