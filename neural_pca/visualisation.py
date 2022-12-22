import sys
sys.path.insert(0,'../')
import os

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seed

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import activation_space
from utils.temperature_wrapper import TemperatureWrapper
from counterfactual import create_results_dir
import data 
import activations
from utils.get_models import get_model

import argparse


def get_latent_space(model, last_layer, target_class, device, batch_size=128, n_workers=8, save_dir=None, load_path=None):
    # Load training data
    train_loader, _ = data.imagenet_subset(target_class, batch_size=batch_size, n_workers=n_workers)

    # Compute latent space model
    act_space = activation_space.ActivationSpace(model, last_layer, target_class, device, train_loader)
    
    if load_path is None:
        act_space.fit()
    else:
        act_space.load(load_path)
    
    # Save 
    if save_dir is not None:
        act_space.save(f'{save_dir}/act_space.npy')
    return act_space


def save_torch_img(img_tensor, path):
    img_arr = img_tensor.permute(1,2,0).cpu().numpy() * 255
    img_arr = img_arr.astype(np.uint8)
    img_pil = Image.fromarray(img_arr)
    img_pil.save(path)


class AlphaWrapper(torch.nn.Module):
    def __init__(self, act_space):
        super(AlphaWrapper, self).__init__()
        self.act_space = act_space
        self.model = act_space.model

    def __call__(self, x):
        act = activations.activations_with_grad(x, self.model, self.act_space._last_layer, grad_enabled=True)
        act = act * self.act_space._last_weights
        alpha = self.act_space._pca_transform(act)
        return alpha


class WeightedActivationPCTarget:
    def __init__(self, pc_dim):
        self.pc_dim = pc_dim

    def __call__(self, model_out):
        return model_out[self.pc_dim]


def visualise_components(class_idx, device, batchsize, n_components=10, load_act_space=False, load_conf=False):
    config = {
        'target_class':class_idx,
        'loss':'obj_full',
        'model':'robust_resnet',
        'device':device,
        'n_pcs': n_components,
        'batch_size':batchsize,
        'n_workers':8,
        'rnd_seed':0,
        'results_dir':'visualisations',
        'load_act_space':load_act_space,
        'criterion':'cf_conf'
    }

    model_id = config['model']
    target_name = data.imagenet_label2class[config['target_class']]
    save_dir = config['results_dir']
    act_space_save_dir = 'act_spaces'

    if save_dir is not None:
        target_cls_str = f'{config["target_class"]:03d}'
        save_dir = create_results_dir(f'{save_dir}/{target_cls_str}_{target_name}')
        act_space_save_dir = create_results_dir(f'{act_space_save_dir}/{target_cls_str}_{target_name}')
        img_dir = create_results_dir(f'{save_dir}/img')
        
    # Set random seed
    seed.set_rand_seed(config['rnd_seed'])

    # Load model
    model, _ = get_model(device, None, model_id)
    
    if model_id == "robust_resnet":
        model = TemperatureWrapper(model)
        last_layer = model.model.model.fc
        gradcam_layers = [model.model.model.layer4[-1]]
    else:
        last_layer = None
        gradcam_layers = None

    # Compute latent space
    if config['load_act_space']:
        load_path = f'{act_space_save_dir}/act_space.npy'
    else:
        load_path = None
    act_space = get_latent_space(model, last_layer, config['target_class'], config['device'], config['batch_size'], config['n_workers'], act_space_save_dir, load_path)

    if not load_conf:
        with torch.no_grad():
            for img, _ in act_space._train_loader:
                out = act_space.model(img.to(config['device']))
                conf = torch.softmax(out, dim=1).cpu().detach().numpy()
                if act_space.confidences_train is None:
                    act_space.confidences_train = conf
                else:
                    act_space.confidences_train = np.concatenate((act_space.confidences_train, conf))

    # Compute counterfactuals on grey backgrounds
    target_pcs = np.arange(config['batch_size'])
    cfs, alpha, prob, pred, objective_vals = act_space.compute_feature_vce(target_pcs, rnd_seed=config['rnd_seed'], loss=config['loss'], eigvec_scale=True, return_losses=True)
    objective_vals = torch.cat(objective_vals)

    # Select top components wrt objective
    if config['criterion'] == 'cf_alpha':
        criterion = objective_vals[:, -1]
    elif config['criterion'] == 'cf_conf':
        criterion = prob[:, config['target_class']]
    elif config['criterion'] == 'cf_alpha_pos':
        pos_idcs = np.where(objective_vals[:, -1] > 0)
        criterion = objective_vals[:, -1] / torch.sum(objective_vals[pos_idcs])
    elif config['criterion'] == 'cf_conf_alpha_pos':
        pos_idcs = np.where(objective_vals[:, -1] > 0)
        alpha_pos = objective_vals[:, -1] / torch.sum(objective_vals[pos_idcs])
        criterion = prob[:,config['target_class']] * alpha_pos
    elif config['criterion'] == 'cf_conf_alpha':
        criterion = objective_vals[:, -1] * prob[:, config['target_class']]
    top_idcs = torch.argsort(criterion, descending=True).cpu().numpy()[:config['n_pcs']]
    
    # Compute maximal activating training images
    n_max_imgs = 5

    cf_cam_greyscales = []
    max_cam_greyscales = {}
    max_imgs = {}
    max_idcs = {}
    for pc_idx in top_idcs:
        max_cam_greyscales[pc_idx] = []
        max_imgs[pc_idx], max_idcs[pc_idx] = act_space.maximizing_train_points(pc_idx, k=n_max_imgs, return_indices=True)

        # Compute GradCAM wrt components
        cam = GradCAM(model=AlphaWrapper(act_space), target_layers=gradcam_layers, use_cuda=False)
        cf_img_targets = [WeightedActivationPCTarget(pc_idx)]
        max_img_targets = [WeightedActivationPCTarget(pc_idx) for _ in range(n_max_imgs)]

        cf_cam_greyscales.append(cam(input_tensor=cfs[pc_idx][None,:].to(config['device']), targets=cf_img_targets).squeeze())
        for max_img in max_imgs[pc_idx]:
            max_cam_greyscales[pc_idx].append(cam(input_tensor=max_img.to(config['device']), targets=max_img_targets).squeeze())
    
    # Plot components
    fontsize = 20
    for i, pc_idx in enumerate(top_idcs):
        fig, ax = plt.subplots(2, 1 + n_max_imgs, figsize=(n_max_imgs*7, 14), constrained_layout=True)
        
        fig.suptitle(f'Class {target_name} - Component {i+1} ({pc_idx} by eigenval)', fontsize=fontsize*2)

        # Feature Attack
        ax[0,0].imshow(cfs[pc_idx].permute(1,2,0))
        ax[0,0].set_title(f'Prediction: {data.imagenet_label2class[pred[pc_idx].item()]}\nConf. {target_name}: {prob[pc_idx, config["target_class"]]:.4f}\nAlpha: {alpha[pc_idx, pc_idx]:.4f}\nCriterion: {criterion[pc_idx]:.4f}', fontsize=fontsize)
        ax[0,0].set_ylabel(f'PC {pc_idx}', fontsize=fontsize)
        ax[0,0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)

        # Feature Attack - GradCAM
        cf_cam = show_cam_on_image(cfs[pc_idx].permute(1,2,0).cpu().numpy(), cf_cam_greyscales[i], use_rgb=True)
        ax[1,0].imshow(cf_cam)
        ax[1,0].set_ylabel(f'GradCAM (alpha)', fontsize=fontsize)

        for j, (max_img, max_idx) in enumerate(zip(max_imgs[pc_idx], max_idcs[pc_idx])):
            
            # Max. activating training image
            ax[0,j+1].imshow(max_img.permute(1,2,0))
            ax[0,j+1].set_title(f'Max. act. train - {j}\nConf. {target_name}: {act_space.confidences_train[max_idx, config["target_class"]]:.4f}\nAlpha: {act_space.pca_train[max_idx, pc_idx]:.4f}', fontsize=fontsize)
            ax[0,j+1].axis('off')

            # Max. activating training image - GradCAM
            max_cam = show_cam_on_image(max_img.permute(1,2,0).cpu().numpy(), max_cam_greyscales[pc_idx][j], use_rgb=True)
            ax[1,j+1].imshow(max_cam)
            ax[1,j+1].axis('off')
        
        plt.savefig(f'{save_dir}/{i+1}_visualize_component_{pc_idx}.png')
        plt.close()

    if save_dir is not None:
        for pc_idx in top_idcs:
            # save NPFVs
            save_torch_img(cfs[pc_idx], f'{img_dir}/cf_pc_{pc_idx}.png')
            
            cfs_data = {
                'top_idcs':top_idcs,
                'alpha':alpha[top_idcs],
                'prob':prob[top_idcs],
                'pred':pred[top_idcs],
                'obj_vals':objective_vals[top_idcs]
            }

            np.save(f'{save_dir}/cfs_data.npy', cfs_data)

            # save max. training imgs
            for count, max_img in enumerate(max_imgs[pc_idx]):
                save_torch_img(max_img, f'{img_dir}/act_pc_{pc_idx}_max_{count}.png')
    
def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--class_idx', default=0, type=int, 
                    help='ImageNet class index.')
    parser.add_argument('--bs', default=16, type=int, 
                    help='Batchsize.')
    parser.add_argument('--gpu', default=0, type=int, 
                    help='GPU index.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    class_idx = args.class_idx    
    bs = args.bs
    
    visualize_components(class_idx, device, bs)

