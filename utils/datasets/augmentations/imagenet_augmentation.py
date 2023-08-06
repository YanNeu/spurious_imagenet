from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from .autoaugment import ImageNetPolicy, CIFAR10Policy
from .cutout import Cutout
from .cifar_augmentation import CIFAR10_mean
from PIL import Image

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


ImageNet_mean_int = ( int( 255 * 0.485), int(255 * 0.456), int(255 * 0.406))

default_interpolation = InterpolationMode.BICUBIC
def get_imageNet_augmentation(type='default', out_size=224, interpolation=default_interpolation, config_dict=None):
    if type == 'none' or type is None:
        transform_list = [
            transforms.Resize((out_size, out_size), interpolation=interpolation),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        return transform
    elif type == 'test' or type is None:
        transform_list = [
            transforms.Resize(int(256/224 * out_size), interpolation=interpolation),
            transforms.CenterCrop(out_size),
        ]
    elif type == 'no_crop':
        transform_list = [
            transforms.Resize(out_size, interpolation=interpolation),
        ]
    elif type == 'default':
        transform_list = [
            transforms.transforms.RandomResizedCrop(out_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Output out_size'] = out_size

    return transform