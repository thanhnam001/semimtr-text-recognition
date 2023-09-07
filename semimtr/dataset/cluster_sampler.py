from tqdm import tqdm
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler

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