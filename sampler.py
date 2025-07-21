import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from torch.utils.data import Sampler, Dataset, RandomSampler


class GroupedPrefixSampler(Sampler):
    def __init__(self, data_source, group_fn=None, shuffle=True):
        """
        Args:
            data_source: a dataset object with access to filenames
            group_fn: a function that extracts the group key from a filename
            shuffle: whether to shuffle groups and elements
        """
        self.data_source = data_source
        self.shuffle = shuffle

        self.group_map = defaultdict(list)  # {prefix: [indices]}
        fnames = [x[0].split('/')[-1] for x in data_source.samples]

        image_ids = [int(fname.split('_')[0]) for fname in fnames]
        annotation_ids = [int(fname.split('_')[1].split('.')[0]) for fname in fnames]
        for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            self.group_map[image_id].append(idx)

        self.groups = list(self.group_map.values())


    def __iter__(self):
        groups = deepcopy(self.groups)
        self.pairs = []
        tot=0
        while tot < len(self.data_source.samples):
            random.shuffle(groups)
            for group in groups:
                random.shuffle(group)
            for i, group in enumerate(groups):
                    if len(group) == 0:
                        continue
                    sub_group = groups[i][:np.random.choice(list(range(1,len(groups[i])+1)))]
                   
                    for idx in sub_group:
                        groups[i].remove(idx)
                        tot += 1
                    self.pairs += [sub_group]
        random.shuffle(self.pairs)
        self.idxs = [x for group in self.pairs for x in group]
        return iter(self.idxs)


    def __len__(self):
        return sum(len(g) for g in self.groups)