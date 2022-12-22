import torch
import numpy as np
from captum import attr as cattr

class SpuriousProjectionWrapper(torch.nn.Module):

    def __init__(self, model,  spurious_components, weighted_act_means, eigenvectors, last_layer, device):
        super(SpuriousProjectionWrapper, self).__init__()
        self.spurious_components = spurious_components
        self.spurious_classes = list(self.spurious_components.keys())
        self.model = model
        self.eigenvectors = eigenvectors[self.spurious_classes].to(device)
        self.w_act_means = weighted_act_means[self.spurious_classes].to(device)
        self.last_layer = last_layer
        self._layer_activations = cattr.LayerActivation(model, last_layer)
        self.device = device

    def __call__(self, x, debug=False):
        n_in = x.shape[0]
        n_spur = len(self.spurious_classes)
        n_comp = self.eigenvectors.shape[1]
        activations = self._layer_activations.attribute(x, attribute_to_layer_input=True)
        mean_w_act = torch.sum(self.w_act_means, dim=1)

        weighted_activations = activations[:, None, :] * self.last_layer.weight[self.spurious_classes]
        w_act_pca = weighted_activations - self.w_act_means
        alphas = torch.zeros((n_in, n_spur, n_comp), device=self.device) 
        for spur_idx, class_idx in enumerate(self.spurious_classes):
            alphas[:, spur_idx, :] = w_act_pca[:, spur_idx, :]@self.eigenvectors[spur_idx]
            alphas[:, spur_idx, :] *= torch.sum(self.eigenvectors[spur_idx], dim=0).to(self.device)
            for comp in self.spurious_components[class_idx]:
                idcs_zero = alphas[:, spur_idx, comp] > 0
                alphas[idcs_zero, spur_idx, comp] = 0
        alpha_sum = torch.sum(alphas, dim=2)
        logit_proj = alpha_sum.to(self.device) + mean_w_act
        logits = activations@self.last_layer.weight.T
        logits_debug = logits.clone()
        logits[:, self.spurious_classes] = logit_proj
        logits += self.last_layer.bias
        if debug:
            return logits, logits_debug
        else:
            return logits


def get_last_layer(model):
    model_rec = model
    depth = 0
    while True:
        if depth > 10:
            raise ValueError('Can not determine head of model')
        elif hasattr(model_rec, 'fc'):
            fc = model_rec.fc
            return fc
        elif hasattr(model_rec, 'head'):
            model_rec = model_rec.head
        else:
            model_rec = model.model
            depth += 1


def wrap_model(model, device):
    spurious_components = np.load('spurious_projection_files/spurious_components.npy', allow_pickle=True)[()]
    means = torch.load('spurious_projection_files/all_means.npy')
    eigenvectors = torch.load('spurious_projection_files/all_eigenvectors.npy')
    last_layer = get_last_layer(model)
    model_proj = SpuriousProjectionWrapper(
        model, spurious_components, means, eigenvectors, last_layer, device)
    return model_proj
