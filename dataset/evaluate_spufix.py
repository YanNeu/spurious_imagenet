import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import argparse

from spurious_score import eval_spurious_score, get_model, get_loaders

from utils.get_last_layer import get_last_layer
from spufix.matching_directions import compute_activations, find_directions
from spufix.spufix_wrapper import load_spufix_model 

SPUR_COMPS_PATH = "../dataset/spurious_components.npy"

def get_devices(gpus):
    if len(gpus) == 0:
        device_ids = None
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(gpus) == 1:
        device_ids = None
        device = torch.device('cuda:' + str(gpus[0]))
    else:
        device_ids = [int(i) for i in gpus]
        device = torch.device('cuda:' + str(min(device_ids)))
    return device, device_ids

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--bs', default=64, type=int,
                    help='batch size.')
    parser.add_argument('--model_id', type=str, default='robust_resnet')
    parser.add_argument('--load_act', action='store_true')
    parser.add_argument('--load_direction', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device, device_ids = get_devices(args.gpu)
    bs = args.bs

    spurious_components = np.load(SPUR_COMPS_PATH, allow_pickle=True)[()]
    classes = list(spurious_components.keys())

    # Model
    """
    evaluate pre-trained models from timm :
        -just set the model_name accordingly
        - make sure the model's architecture is implemented in utils.get_last_layer.py
    
    evaluate your own models:
        - replace the get_model function
        - model should include a normalization wrapper (see utils.model_normalization.py)
        - img_size format (3, <size>, <size>)
        - make sure the model's architecture is implemented in utils.get_last_layer.py
    """

    model_name = args.model_id
    model, img_size = get_model(device, device_ids, model_name)
    
    multi_gpu  = not device_ids is None
    last_layer = get_last_layer(model_name, model, multi_gpu)
    
    bs = args.bs
    if not args.load_direction:
        # compute penultimate layer features on training images
        pbar = tqdm(classes)
        if not args.load_act:
            for class_idx in pbar:
                pbar.set_description(f"Compute penultimate layer features for class {class_idx:3d}")
                compute_activations(model_name, model, last_layer, class_idx, device, img_size, bs)

        # match latent space direction to alpha values on training images
        find_directions(model_name, classes, spurious_components, device)

    # load transfer spufix model
    spufix_model, img_size = load_spufix_model(model_name, model, last_layer, img_size, classes, device)
    spufix_model_name = f'{model_name}_spufix'

    # load datasets
    spurious_loader, in_subset_loader = get_loaders(img_size, bs)

    eval_spurious_score(spufix_model, spufix_model_name, device, spurious_loader, in_subset_loader)

