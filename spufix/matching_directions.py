import sys
sys.path.insert(0,'../')

import numpy as np

import argparse
import torch

import neural_pca.data as data
from neural_pca.counterfactual import create_results_dir
from neural_pca.activation_space import ActivationSpace

from utils.get_models import get_model
from utils.get_last_layer import get_last_layer

from tqdm import tqdm

TRAIN_ALPHAS_DIR = "../neural_pca/spurious_alphas_train"
SPUR_COMPS_PATH = "../dataset/spurious_components.npy"
ACT_SPACE_DIR = "../neural_pca/act_spaces"
SAVE_DIR = "../spufix/matched_directions" 

def max_cov(alpha_train, act_train, act_mean):
    # centered weighted training activations
    centered_act = act_train - act_mean
    
    alpha_act = torch.sum(alpha_train[:, None] * centered_act, dim=0)
    cov = torch.linalg.norm(alpha_act, ord=2)
    max_cov_direction = alpha_act/cov

    return max_cov_direction, cov


def compute_activations(model_id, model, last_layer, target_class, device, img_size, batch_size=128, n_workers=8):
    # Load training data
    train_loader = data.imagenet_subset(target_class, batch_size=batch_size, img_size=img_size[1], n_workers=n_workers, only_train=True)
    
    # Compute latent space model
    act_space = ActivationSpace(model, last_layer, target_class, device, train_loader)
    act_space.fit(only_act=True)

    class_str = f"{target_class:03d}_{data.imagenet_label2class[target_class]}"
    act_space_dir = create_results_dir(f"{ACT_SPACE_DIR}/act_spaces_{model_id}/{class_str}")
    act_space.save(f'{act_space_dir}/act_space.npy')


def find_directions(model_id, spurious_classes, spurious_components, device, obj='max_cov'):
    save_dir = create_results_dir(f"{SAVE_DIR}/{model_id}_{obj}")
    act_space_dir = f"{ACT_SPACE_DIR}/act_spaces_{model_id}"

    pbar = tqdm(spurious_classes)
    for class_idx in pbar:
        pbar.set_description(f"Matching direction for class {class_idx:3d}")
        class_str = f"{class_idx:03d}_{data.imagenet_label2class[class_idx]}"

        matched_directions = {}
        obj_values = {}

        # load spurious training alphas
        spurious_alphas = torch.load(f"{TRAIN_ALPHAS_DIR}/spurious_alphas_train_{class_idx:03d}.pt")
        
        # load weighted activations and mean
        act_space = np.load(f"{act_space_dir}/{class_str}/act_space.npy", allow_pickle=True)[()]
        weighted_activations_train = act_space['act_train']
        weighted_activations_train_mean = act_space['pca_mean']

        if not torch.is_tensor(weighted_activations_train):
            weighted_activations_train = torch.from_numpy(weighted_activations_train)
        if not torch.is_tensor(weighted_activations_train_mean):
            weighted_activations_train_mean = torch.from_numpy(weighted_activations_train_mean)
        
        for comp_idx in spurious_components[class_idx]:
            direction, obj_value = max_cov(
                spurious_alphas[:, comp_idx].to(device),
                weighted_activations_train.to(device),
                weighted_activations_train_mean.to(device))
            
            matched_directions[comp_idx] = direction.detach().cpu()
            obj_values[comp_idx] = obj_value.detach().cpu()
           
        torch.save(matched_directions, f"{save_dir}/matched_spurious_directions_{obj}_{class_idx:03d}.pt")
        torch.save(obj_values, f"{save_dir}/obj_values_{obj}_{class_idx:03d}.pt")


def compute_transfer_spufix(model_id, device, device_ids, batch_size=32, load_activations=False):
    spurious_components = np.load(SPUR_COMPS_PATH, allow_pickle=True)[()]
    classes = spurious_components.keys()

    if not load_activations:
        pbar = tqdm(classes)

        for class_idx in pbar:
            pbar.set_description(f"Compute penultimate layer features for class {class_idx:3d}")
            model, img_size = get_model(device, device_ids, model_id)
            multi_gpu  = not device_ids is None
            last_layer = get_last_layer(model_id, model, multi_gpu)

            compute_activations(model_id, model, last_layer, class_idx, device, img_size, batch_size)
        
    find_directions(model_id, classes, spurious_components, device)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--bs', default=64 , type=int, 
                    help='Batchsize.')
    parser.add_argument('--gpu','--list', nargs='+', default=[0],
                    help='GPU indices')
    parser.add_argument('--model_id', default="resnet50", type=str)
    parser.add_argument('--load_act', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    

    if len(args.gpu) == 0:
        device_ids = None
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(args.gpu) == 1:
        device_ids = None
        device = torch.device('cuda:' + str(args.gpu[0]))
    else:
        device_ids = [int(i) for i in args.gpu]
        device = torch.device('cuda:' + str(min(device_ids)))


    model_id = args.model_id
    compute_transfer_spufix(model_id, device, device_ids, batch_size=args.bs, load_activations=args.load_act)