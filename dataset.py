import os
import itertools
import numpy as np
import pandas as pd

from glob import glob
from PIL import Image

import torch
from torch.utils.data import Sampler, Dataset, RandomSampler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandAugment

import pytorch_lightning as pl

from sampler import GroupedPrefixSampler



class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4, img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size


        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )


        self.train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomAffine(degrees=(-10,10)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            #transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_set = CustomImageFolder(os.path.join(self.data_dir, 'train'), transform=self.train_transform)
        self.val_set   = CustomImageFolder(os.path.join(self.data_dir, 'val'), transform=self.test_transform)

        self.class_names = self.train_set.classes

    def setup_samplers(self, stage=None):
        self.train_sampler = GroupedPrefixSampler(self.train_set)
        self.val_sampler = GroupedPrefixSampler(self.val_set, shuffle=False)
        #self.train_sampler = RandomSampler(self.train_set)
        #self.val_sampler = RandomSampler(self.val_set)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.val_sampler)


class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, custom_arg=None):
        """
        CustomImageFolder extends ImageFolder to allow additional processing.

        Args:
            root (str): Root directory path.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            custom_arg (any, optional): Example custom argument to show how to extend the init.
        """
        # self.edge_indices = self.get_edge_indices()  # store your custom arg or do something with it
        self.root = root

        self.image_paths = glob(os.path.join(self.root, '*', '*.png'))
        class_names = np.unique([x.split('/')[-2] for x in self.image_paths]).tolist()
        self.classes = sorted(class_names)
        self.class_to_idx = {name : i for i, name in enumerate(self.classes)}
        self.samples = self.get_samples()
        self.image_ids = np.array([int(x[0].split('/')[-1].split('_')[0]) for x in self.samples])
        self.annotation_ids = np.array([int(x[0].split('_')[-1].split('.')[0]) for x in self.samples])
        self.transform = transform
        self.target_transform = target_transform

    def get_samples(self):
        classes = [x.split('/')[-2] for x in self.image_paths]
        return [(path, self.class_to_idx[_class]) for path, _class in zip(self.image_paths, classes)]

    def get_edge_indices(self):
        image_ids = [int(x[0].split('/')[-1].split('_')[0]) for x in self.samples]
        annotation_ids = [int(x[0].split('_')[-1].split('.')[0]) for x in self.samples]
        df = pd.DataFrame(np.array([image_ids, annotation_ids]).T, columns=['image_id','annotation_id'])
        df = df.sort_values('annotation_id')

        edge_index = []
        for name, group in df.groupby('image_id'):
            ids = group['annotation_id'].values - 1
            edge_index += list(itertools.product(ids,ids))
        edge_index = [x for x in edge_index if x[0]!=x[1]]
        edge_index = np.array(edge_index)

        edge_indices = {}
        for i in range(max(annotation_ids)):
            edge_indices[i] = [x[1] for x in edge_index if x[0] == i]

        return edge_indices


    def pil_loader(self, path) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        """
        Custom __getitem__ to extend or modify data returned for each item.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) or a custom tuple depending on your use case.
        """
        path, target = self.samples[index]
        sample = self.pil_loader(path)
        img_size = np.mean(sample.size) / 1000

        # Apply default transforms if present
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Modify the return if needed
        # For example, you could return the image path as well
        return {'x':sample, 'y':target, 'image_id':self.image_ids[index], 'path':path, 'img_size':img_size}  # or (sample, target, path) or anything else
    
    def __len__(self):
        return len(self.samples)

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_ids = np.array([int(x.split('/')[-1].split('_')[0]) for x in self.image_paths])
        self.annotation_ids = np.array([int(x.split('_')[-1].split('.')[0]) for x in self.image_paths])

        self.samples = [(path, '') for path in self.image_paths]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        img_size = np.mean(image.size) / 1000
        if self.transform:
            image = self.transform(image)
        return {'x':image, 'path':path, 'image_id':self.image_ids[idx], 'img_size': img_size}  # we return the path so we can identify predictions
