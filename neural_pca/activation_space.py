import torch
import numpy as np
from captum import attr as cattr

from neural_pca.counterfactual import generate_feature_counterfactuals
from neural_pca.data import imagenet_label2class
from neural_pca.pca import compute_pca

class  ActivationSpace():

    def __init__(self, model, last_layer, target_idx, device, train_loader=None):
        self.target = target_idx
        self.target_name = imagenet_label2class[self.target] 
        
        self.device=device        
        self.model = model.to(self.device)

        self._last_layer = last_layer 
        self._layer_activations = cattr.LayerActivation(model, last_layer)
        self._train_loader = train_loader
        self._last_weights = last_layer.weight.data[self.target]
        
    def fit(self, eigvec_scale=True):
        if self._train_loader is None:
            raise Exception('Missing train_loader. Use AcitvationSpace.load() or init with train_loader.')
        
        # get activations
        print('Computing activations\n')
        self.activations_train = self._compute_training_activations()

        # compute pca
        print('Computing principal components\n')
        _, self.eigenvectors, pca_mean = compute_pca(self.activations_train.T.cpu().numpy())
        self.eigenvectors = torch.tensor(self.eigenvectors, dtype=torch.float32, device=self.device)
        self.pca_mean = torch.tensor(pca_mean.T, device=self.device)
        
        # transform training points
        self.pca_train = self._pca_transform(self.activations_train, eigvec_scale=eigvec_scale)
        
    def transform(self, images, eigvec_scale=True):
        act = self._compute_activations(images)
        return self._pca_transform(act, eigvec_scale=eigvec_scale)

    def compute_feature_vce(self, pca_dims, background_color=[0.5,0.5,0.5], norm='L2', eps=30, steps=200, rnd_seed=0, loss='obj', reg_other=1.0, eigvec_scale=True, return_losses=False, minimize=False, minimize_abs=False):
        assert len(background_color) == 3
        assert background_color[0] <= 1 and background_color[0] >= 0
        assert background_color[1] <= 1 and background_color[1] >= 0
        assert background_color[2] <= 1 and background_color[2] >= 0

        n_targets = len(pca_dims)
        
        # create backgrounds
        backgrounds = torch.zeros((n_targets, 3, 224, 224), device=self.device)
        backgrounds += torch.tensor(background_color, device=self.device)[:, None, None]
        
        # generate counterfactuals
        print('Computing feature counterfactuals\n')
        perturbation_targets = torch.tensor(pca_dims, dtype=torch.long)
        if return_losses:
            cfs, losses = generate_feature_counterfactuals(
                model=self.model,
                images=backgrounds,
                last_layer=self._last_layer,
                eigenvectors=self.eigenvectors,
                target_cls=self.target,
                norm=norm,
                eps=eps,
                steps=steps,
                perturbation_targets=perturbation_targets,
                device=self.device,
                seed=rnd_seed,
                loss=loss,
                reg_other=reg_other,
                return_losses=True,
                minimize=minimize,
                minimize_abs=minimize_abs,
            )
        else:
            cfs = generate_feature_counterfactuals(
                model=self.model,
                images=backgrounds,
                last_layer=self._last_layer,
                eigenvectors=self.eigenvectors,
                target_cls=self.target,
                norm=norm,
                eps=eps,
                steps=steps,
                perturbation_targets=perturbation_targets,
                device=self.device,
                seed=rnd_seed,
                loss=loss,
                reg_other=reg_other,
                minimize=minimize,
                minimize_abs=minimize_abs
            )

        with torch.no_grad():
            out = self.model(cfs)
            prob = torch.softmax(out, dim=1).cpu().detach()
            pred = torch.max(out, dim=1)[1].cpu().detach()

            act = self._compute_activations(cfs)
            pca_cfs = self._pca_transform(act, eigvec_scale)
        
        ret = (cfs.cpu().detach(), pca_cfs, prob, pred)        
        if return_losses:
            ret = (*ret, losses)
        return ret

    def top_features(self, k=10, order='eigenvalues', background_color=[0.5, 0.5, 0.5], eps=30, steps=200, rnd_seed=0, pc_batch_size=128, loss='obj', reg_other=1.0, eigvec_scale=True):
        assert order in ['eigenvalues', 'confidence', 'eigenvector-l1', 'max-logit-contribution']    
        if order == 'eigenvalues':
            pca_dims = list(range(k))
        elif order == 'confidence':
            pca_dims = list(range(pc_batch_size))        
        elif order == 'eigenvector-l1':
            idcs_sorted = np.argsort(np.sum(self.eigenvectors, axis=0))
            pca_dims = np.flip(idcs_sorted[-k:]).copy()
        elif order == 'max-logit-contribution':
            max_contributions = np.max(self.pca_train, axis=0)
            idcs_sorted = np.argsort(max_contributions)
            pca_dims = np.flip(idcs_sorted[-k:]).copy()
        
        cfs, pca_cfs, prob, pred = self.compute_feature_vce(
                pca_dims=pca_dims, 
                background_color=background_color, 
                eps=eps, 
                steps=steps, 
                rnd_seed=rnd_seed,
                loss=loss,
                reg_other=reg_other,
                eigvec_scale=eigvec_scale)
        pca_dims = np.array(pca_dims)
        if order in ['eigenvalues', 'eigenvector-l1', 'max-logit-contribution']:
            return cfs, pca_cfs, prob, pred, pca_dims
        elif order == 'confidence':
            sorted_idcs = torch.argsort(prob[:, self.target], descending=True)
            return cfs[sorted_idcs][:k], pca_cfs[sorted_idcs][:k], prob[sorted_idcs][:k], pred[sorted_idcs][:k], pca_dims[sorted_idcs][:k]
    
    def knn(self, x, k=5, return_indices=False):
        diffs = np.linalg.norm(self.pca_train - x, axis=1)
        knn_idcs = np.argsort(diffs)[:k]
        neighbours = []
        for idx in knn_idcs:
            neighbours.append(self._train_loader.dataset[idx][0])
        if return_indices:
            return neighbours, knn_idcs
        return neighbours
    
    def maximizing_train_points(self, pca_dim, k=5, return_indices=False, order='alpha'):
        if order == 'alpha':
            return self._maximizing_train_points_alpha(pca_dim, k=k, return_indices=return_indices)
        elif order == 'alpha_conf':
            return self._maximizing_train_points_alpha_conf(pca_dim, k=k, return_indices=return_indices)
    
    def _maximizing_train_points_alpha_conf(self, pca_dim, k=5, return_indices=False):
        representation = self.pca_train
        logits = self.activations_train@self._last_layer.weight.T + self._last_layer.bias
        self.objective = representation[:, pca_dim] - torch.log(torch.sum(torch.exp(logits))) 
        sorted_idcs = torch.argsort(self.objective, descending=True)
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self._train_loader.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        return max_images

    def _maximizing_train_points_alpha(self, pca_dim, k=5, return_indices=False):
        representation = self.pca_train
        self.objective = representation[:, pca_dim]
        sorted_idcs = torch.argsort(representation[:, pca_dim], descending=True)
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self._train_loader.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        return max_images

    def _compute_training_activations(self):
        act_train = None
        self.confidences_train = None
        for imgs, _ in self._train_loader:
            act = self._compute_activations(imgs)

            out = self.model(imgs.to(self.device))
            prob = torch.softmax(out, dim=1).cpu().detach().numpy()
            pred = torch.max(out, dim=1)[1].cpu().detach().numpy()
            
            if act_train is None:
                act_train = act
                self.confidences_train = prob
                self.predictions_train = pred
            else:
                act_train = torch.cat((act_train, act))
                self.confidences_train = np.concatenate((self.confidences_train, prob))
                self.predictions_train = np.concatenate((self.predictions_train, pred))
        return act_train

    def _compute_activations(self, images):
        if len(images.shape) == 3:
            images = images[None, :]
        act = self._layer_activations.attribute(images.to(self.device), attribute_to_layer_input=True)
        act = act.squeeze()            
        return act * self._last_weights
        
    def _pca_transform(self, activations, eigvec_scale=True):
        activations -= self.pca_mean
        act_pca = activations@self.eigenvectors 
        if eigvec_scale:
            act_pca = act_pca * torch.sum(self.eigenvectors, dim=0)
        return act_pca

    def save(self, fpath='actspace.npy'):
        save_dict = {
            'act_train':self.activations_train,
            'eigenvectors':self.eigenvectors,
            'alpha_train':self.pca_train,
            'pca_mean':self.pca_mean,
        }
        np.save(fpath, save_dict)

    def load(self, fpath='actspace.npy'):
        load_dict = np.load(fpath, allow_pickle=True)[()]
        self.activations_train = load_dict['act_train']
        self.eigenvectors = load_dict['eigenvectors'].to(self.device)
        self.pca_train = load_dict['alpha_train']
        self.pca_mean = load_dict['pca_mean']