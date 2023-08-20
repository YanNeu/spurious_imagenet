import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import default_loader

from utils.datasets.imagenet import get_imagenet_path
from utils.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from torchvision.transforms.functional import InterpolationMode


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

        self.internal_idcs = torch.argsort(torch.LongTensor(targets))
        #print(f'SpuriousImageNet - {len(subdirs)} classes - {len(imgs)} images')
        self.transform = transform
        self.imgs = imgs
        self.targets = targets
        self.components = components
        self.included_classes = list(set(list(self.targets)))

    def get_class_component_pairings(self):
        class_components = {}
        for idx in self.internal_idcs:
            idx = idx.item()
            class_components[(self.targets[idx], self.components[idx])] = None

        return list(class_components.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        internal_idx = self.internal_idcs[idx].item()
        img = default_loader((self.imgs[internal_idx]))
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[internal_idx]
        return img, label


def get_spurious_datasets(dataset_dir, img_size=224, interpolation=InterpolationMode.BICUBIC, bs=128, num_workers=1):
    transform = get_imageNet_augmentation(type='no_crop', out_size=img_size, interpolation=interpolation)
    dataset = SpuriousDataset(dataset_dir, transform)
    loader = DataLoader(dataset, batch_size=bs,  shuffle=False, num_workers=num_workers)
    return loader


def get_spurious_datasets_crop_resize(dataset_dir, img_size=224, interpolation=InterpolationMode.BICUBIC,
                                      bs=128, num_workers=1):
    transform = transforms.Compose([transforms.CenterCrop(int(0.875 * 420)),
                                    transforms.Resize(img_size, interpolation=interpolation),
                                    transforms.ToTensor()])
    dataset = SpuriousDataset(dataset_dir, transform)
    loader = DataLoader(dataset, batch_size=bs,  shuffle=False, num_workers=num_workers)
    return loader


def get_imagenet_matching_subset(included_classes, img_size=224, interpolation=InterpolationMode.BICUBIC, bs=128, num_workers=1):
    transform = get_imageNet_augmentation(type='test', out_size=img_size, interpolation=interpolation)
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
