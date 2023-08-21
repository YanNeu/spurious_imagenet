"""
Based on https://github.com/valentyn1boreiko/SVCEs_code by Valentyn Boreiko.
"""
import sys
sys.path.append('../')

import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from neural_pca.adversarial_attacks.act_apgd import ActivationAPGDAttack
import neural_pca.data

def generate_feature_counterfactuals(
    model,
    images,
    last_layer,
    eigenvectors,
    target_cls,
    norm,
    eps,
    steps,
    perturbation_targets,
    device,
    seed,
    n_restarts=1,
    x_init=None,
    reg_other=1.0,
    result_path=None,
    return_losses=False,
    loss='obj',
    ica_components=None,
    ica_mean=None,
    minimize=False,
    minimize_abs=False):
    """
    Args:
        model: Pytorch model, This model is used to generate the counterfactuals.
        images: torch.Tensor, N x C x H x W, Original images
        last_layer: torch.nn.module last layer of the model
        eigenvectors: corresponding to principal componentg in weighted last layer activation space
        target_cls: determines which weights are used for the weighted activations
        n_classes: int, number of classes in the dataset
        norm: str in ["L2", "L1", "Linf"], norm used for the adversarial attacks
        eps: float, epsilon used for the adversarial attacks
        steps: int, number of steps used for the adversarial attacks
        pgd: str, default "apgd", optimization procedure used for the adversarial attacks
        perturbation_targets: torch.Tensor(dtype=torch.long), N x N_targets, contains N_targets
            perturbation targets (indices of principal components of weighted last layer activations) 
            for each image in images
        device: torch.device, CUDA device used for the generation
        seed: integer, random seed
        momentum: float, momentum used for the adversarial attacks
        stepsize: float, stepsize used for the adversarial attacks
        n_restarts: integer, number of restarts used in the adversarial attack (right now only supported for AFW and APGDs)
        x_init: torch.Tensor, N x C x H x W, Initialization for the adversarial perturbation. If 'None', the perturbations are initialized randomly.
        normalize_gradient: bool, flag that determines whether gradients are normalized
        result_path: str, Path for saving the counterfactuals. If 'None', the counterfactuals are returned instead.
        return_confidences: bool, If True, the confidence and loss values are returned as well (only for APGD)
    """
    assert loss in ['obj', 'obj_full', 'ce', 'ce_abs', 'log_nll', 'soft_log_nll', 'min_other', 'min_other_eig', 'max_comp_conf']
    
    bs = images.shape[0]
    if perturbation_targets.shape[0] > bs:
        perturbation_targets = perturbation_targets[:bs]
    
    with torch.no_grad():
        model.to(device)
        images = images.to(device)
        perturbation_targets = perturbation_targets.to(device)

        if result_path is not None:
            torch.save(images, result_path + "/original_images.pt")
        
        adv_attack = ActivationAPGDAttack(
            model, 
            eps=eps, 
            n_iter=steps, 
            norm=norm, 
            loss=loss, 
            n_restarts=n_restarts, 
            verbose=False, 
            seed=seed,
            last_layer=last_layer,
            eigenvecs=eigenvectors,
            target_cls=target_cls,
            device=device,
            reg_other=reg_other,
            ica_components=ica_components,
            ica_mean=ica_mean,
            minimize=minimize,
            minimize_abs=minimize_abs)

        losses = [] if return_losses else None
        cfs = {}
        if return_losses:
            adv_samples, loss = adv_attack.perturb(
                images,
                perturbation_targets, 
                best_loss=True,
                x_init=x_init)
            adv_samples, loss = adv_samples.detach(), loss.detach()
            losses.append(loss)
        else:
            adv_samples =  adv_attack.perturb(
                images,
                perturbation_targets, 
                best_loss=True,
                x_init=x_init)[0].detach()
            
        if result_path is not None:
            path = result_path + f"/adv_samples_{norm}_eps_{eps}_target_{target_cls}"
            torch.save(adv_samples.detach(), path)
        else:
            cfs = adv_samples.detach()

    if result_path is None:
        if return_losses:
            return cfs, losses
        return cfs

def compute_diff_image(a,b, filepath=None):
    diff = (a - b).sum(2)
    min_diff_pixels = diff.min()
    max_diff_pixels = diff.max()
    min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
    max_diff_pixels = -min_diff_pixels
    diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
    cm = plt.get_cmap('seismic')
    colored_image = cm(diff_scaled.numpy())
    pil_img = Image.fromarray(np.uint8(colored_image * 255.))
    if not filepath is None:
        pil_img.save(filepath)
    return pil_img

def compute_diff_act(a,b, filepath=None):
    diff = a - b
    min_diff_pixels = diff.min()
    max_diff_pixels = diff.max()
    min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
    max_diff_pixels = -min_diff_pixels
    diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
    cm = plt.get_cmap('seismic')
    if type(diff_scaled) != np.ndarray:
        diff_scaled = diff_scaled.numpy()
    colored_image = cm(diff_scaled)
    return colored_image


def imagenet_confidence_all(model, device, fpath='results/pipeline/all_conf', batch_size=128, n_workers=8):
    for target_class in range(1000):
        confidences = []
        with torch.no_grad():
            train_loader, _ = data.imagenet_subset(target_class, batch_size=batch_size, n_workers=n_workers)

            for batch_idx, (img, lab) in enumerate(train_loader):
                out = model(img.to(device))
                prob = torch.softmax(out, dim=1).cpu().detach().numpy()

                confidences.append(prob)
            
        confidences = np.concatenate(confidences, axis=0)
        np.save(f'{fpath}/conf_class_{target_class}.npy', confidences)


def select_start_class(target_class, k=1, fpath='results/pipeline/all_conf', verbose=False, only_geo=False):
    target_conf_means = []
    candidates = [970, 972, 973, 974, 975, 976, 977, 978, 979, 980] if only_geo else range(1000)
    for in_class in candidates:
        confidences = np.load(f'{fpath}/conf_class_{in_class}.npy')
        target_conf_means.append(np.mean(confidences[:, target_class]))

    target_conf_means = np.array(target_conf_means)
    sorted_idcs = np.flip(np.argsort(target_conf_means).copy())
    
    if only_geo:
        similar_idcs = np.array(candidates)[sorted_idcs[:k]]
    else:
        if target_class in sorted_idcs[:k]:
            target_flag = False
            for i in range(k):
                if sorted_idcs[i] == target_class:
                    target_flag = True
                if target_flag:
                    sorted_idcs[i] = sorted_idcs[i+1]
        similar_idcs = sorted_idcs[:k]
    if verbose:
        for i, idx in enumerate(similar_idcs):
            print(idx, data.imagenet_label2class[idx], target_conf_means[i])
    return similar_idcs, target_conf_means[sorted_idcs[:k]]


def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path
