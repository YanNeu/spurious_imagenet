import sys
sys.path.append("../")

import torch
import numpy as np
from captum import attr as cattr
from tqdm import tqdm
import argparse

from spufix.matching_directions import compute_activations, find_directions

import neural_pca.data as data 

from utils.get_last_layer import get_last_layer
from utils.get_models import get_model

ACT_SPACE_DIR = "../neural_pca/act_spaces"
MATCHED_DIR_DIR = "../spufix/matched_directions"
SPUR_COMPS_PATH = "../dataset/spurious_components.npy"

def non_orth_basis_proj(directions, point):
    lam = torch.linalg.solve(directions.T@directions, directions.T@point)
    return lam

class SpuFixWrapper(torch.nn.Module):

    def __init__(self, model, spurious_classes, weighted_act_means, matched_directions, last_layer, device, only_pos=True):
        super(SpuFixWrapper, self).__init__()
        self.spurious_classes = spurious_classes
        self.model = model
        self.matched_directions = matched_directions
        self.w_act_means = weighted_act_means.to(device)
        self.last_layer = last_layer
        self._layer_activations = cattr.LayerActivation(model, last_layer)
        self.device = device
        self.only_pos=only_pos

    def __call__(self, x, return_act=False):
        activations = self._layer_activations.attribute(x, attribute_to_layer_input=True)
        weighted_activations = activations[:, None, :] * self.last_layer.weight[self.spurious_classes]
        w_act_centered = weighted_activations.squeeze() - self.w_act_means
        
        logits = activations.squeeze()@self.last_layer.weight.squeeze().T

        for spur_idx, class_idx in enumerate(self.spurious_classes):
            
            if self.matched_directions[class_idx].shape[0] > 1:
                proj = non_orth_basis_proj(self.matched_directions[class_idx].T, w_act_centered[:, spur_idx].T).T
            else:
                proj = w_act_centered[:, spur_idx]@self.matched_directions[class_idx].T

            beta = proj * torch.sum(self.matched_directions[class_idx], dim=1)

            if self.only_pos:
                beta[beta < 0] = 0
                
            logits[:, class_idx] -= torch.sum(beta, dim=-1) 
        logits += self.last_layer.bias
        
        if return_act:
            return logits, activations
        else:
            return logits


def load_directions(class_idx, directions_dir, matching='max_cov'):
    matched_directions = torch.load(f"{directions_dir}/matched_spurious_directions_{matching}_{class_idx:03d}.pt")
    return matched_directions


def collect_matched_directions(model_id, classes, matching='max_cov'):
    directions_dir = f'{MATCHED_DIR_DIR}/{model_id}_{matching}'

    matched_directions = {}
    for class_idx in classes:
        directions = load_directions(class_idx, directions_dir, matching)
        matched_directions[class_idx] = directions

    return matched_directions


def collect_means(model_id, classes, device):
    load_dir = f"{ACT_SPACE_DIR}/act_spaces_{model_id}"

    all_means = []

    for class_idx in classes:
        act_space = np.load(f"{load_dir}/{class_idx:03d}_{data.imagenet_label2class[class_idx]}/act_space.npy", allow_pickle=True)[()]

        means = act_space["pca_mean"]
        if not torch.is_tensor(means):
            means = torch.from_numpy(means)
        all_means.append(means[None, :].to(device))
    
    all_means = torch.cat(all_means)
    return all_means


def load_spufix_model(model_id, model, last_layer, img_size, classes, device, matching="max_cov"):
    matched_directions = collect_matched_directions(model_id, classes, matching)
    means = collect_means(model_id, classes, device)

    stacked_directions = {}
    act_means = {}
    for i, class_idx in enumerate(classes):
        stacked_directions[class_idx] = []
        act_means[class_idx] = means[i]

        for comp_idx in matched_directions[class_idx]:
            stacked_directions[class_idx].append(matched_directions[class_idx][comp_idx][None, :].to(device))
        
        stacked_directions[class_idx] = torch.cat(stacked_directions[class_idx])
    
    spufix_model = SpuFixWrapper(model, classes, means, stacked_directions, last_layer, device)
    
    return spufix_model, img_size


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--bs', default=64 , type=int, 
                    help='Batchsize.')
    parser.add_argument('--gpu','--list', nargs='+', default=[0],
                    help='GPU indices')
    parser.add_argument('--model_id', default="resnet50", type=str)
    parser.add_argument('--load_act', action='store_true', help='load precomputed activations to match directions')
    parser.add_argument('--load_direction', action='store_true', help='load precomputed matched directions')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spurious_components = np.load(SPUR_COMPS_PATH, allow_pickle=True)[()]
    
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

    classes = list(spurious_components.keys())

    # Model
    """
    evaluate pre-trained models from timm :
        just set the model_name accordingly
    
    evaluate your own models:
        - replace the get_model function with your model
        - model should include a normalization wrapper
        - img_size format (3, <size>, <size>)
        - replace the get_last_layer function with the last (linear) layer of your model
    """

    model_id = args.model_id
    model, img_size = get_model(device, device_ids, model_id)
    
    multi_gpu  = not device_ids is None
    last_layer = get_last_layer(model_id, model, multi_gpu)
       
    if not args.load_direction:
         
        # compute penultimate layer features on training images
        pbar = tqdm(classes)
        if not args.load_act:
            for class_idx in pbar:
                pbar.set_description(f"Compute penultimate layer features for class {class_idx:3d}")
                compute_activations(model_id, model, last_layer, class_idx, device, img_size, args.bs)

        # match latent space direction to alpha values on training images
        find_directions(model_id, classes, spurious_components, device)
    
    # load transfer spufix model
    spufix_model, img_size = load_spufix_model(model_id, model, last_layer, img_size, classes, device)