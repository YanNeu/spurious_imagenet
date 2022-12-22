import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import default_loader

from utils.datasets.imagenet import get_imagenet_path
from utils.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation


class SpuriousDataset(Dataset):
    def __init__(self, path, transform):
        imgs = []
        targets = []
        components = []

        subdirs = next(os.walk(path))[1]
        for subdir in subdirs:
            subdir_class = int(subdir.split('_')[2])
            subdir_component = int(subdir.split('_')[-1])
            for file in os.listdir(os.path.join(path, subdir)):
                imgs.append(os.path.join(path, subdir, file))
                targets.append(subdir_class)
                components.append(subdir_component)

        #print(f'SpuriousImageNet - {len(subdirs)} classes - {len(imgs)} images')
        self.transform = transform
        self.imgs = imgs
        self.targets = targets
        self.components = components
        self.included_classes = list(set(list(self.targets)))

    def get_class_component_pairings(self):
        class_components = set()
        for target, component in zip(self.targets, self.components):
            class_components.add((target, component))
        return list(class_components)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader((self.imgs[idx]))
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

def get_spurious_datasets(dataset_dir, img_size=224, bs=128, num_workers=1):
    #transform = get_imageNet_augmentation(type='test', out_size=img_size)
    transform = get_imageNet_augmentation(type='no_crop', out_size=img_size)
    dataset = SpuriousDataset(dataset_dir, transform)
    loader = DataLoader(dataset, batch_size=bs,  shuffle=False, num_workers=num_workers)
    return loader

def get_imagenet_matching_subset(included_classes, img_size=224, bs=128, num_workers=1):
    transform = get_imageNet_augmentation(type='test', out_size=img_size)
    in_path = get_imagenet_path()
    imagenet = ImageNet(in_path, split='val', transform=transform)

    subset_idcs = torch.zeros(len(imagenet), dtype=torch.bool)
    in_targets = torch.LongTensor(imagenet.targets)
    for in_class in included_classes:
        subset_idcs[in_targets == in_class] = 1

    subset_idcs = torch.nonzero(subset_idcs, as_tuple=False).squeeze()
    in_subset = Subset(imagenet, subset_idcs)

    loader = DataLoader(in_subset, batch_size=bs,  shuffle=False, num_workers=num_workers)
    return loader

def get_cropped_imagenet_matching_subset(included_classes, img_size=224, bs=128, num_workers=1):
    transform = get_imageNet_augmentation(type='sanity', out_size=img_size)
    in_path = get_imagenet_path()
    imagenet = ImageNet(in_path, split='val', transform=transform)

    subset_idcs = torch.zeros(len(imagenet), dtype=torch.bool)
    in_targets = torch.LongTensor(imagenet.targets)
    for in_class in included_classes:
        subset_idcs[in_targets == in_class] = 1

    subset_idcs = torch.nonzero(subset_idcs, as_tuple=False).squeeze()
    in_subset = Subset(imagenet, subset_idcs)

    loader = DataLoader(in_subset, batch_size=bs,  shuffle=False, num_workers=num_workers)
    return loader

