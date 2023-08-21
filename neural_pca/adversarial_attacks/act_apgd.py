# Copyright (c) 2020-present,  Anon1
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#
import sys
sys.path.append('../..')
from audioop import bias
from importlib.metadata import requires
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from RATIO_utils.adversarial_attacks.LPIPS_projection import project_onto_LPIPS_ball
import numpy as np
from tqdm import tqdm

import neural_pca.seed as rnd_sd
from neural_pca.activations import activations_with_grad

from .other_utils import L0_norm, L1_norm, L2_norm

norms_dict_torch = {'L1': 1,
                    'L2': 2,
                    'Linf': np.float('inf'),
                    'LPIPS': None,
                    'FeatureDist': 'FeatureDist',
                    'MS_SSIM_L1': 'MS_SSIM_L1'}

def float_(x):
    try:
       return float(x)
    except:
        return x

get_norm = lambda norm_name: norms_dict_torch.get(norm_name, (lambda x: x)(float_(norm_name)))


def maxlin(x_orig, w_orig, eps, p, verbose=True):
    ''' solves the optimization problem, for x in [0, 1]^d and p > 1,

    max <w, delta> s.th. ||delta||_p <= eps, x + delta \in [0, 1]^d
    '''
    bs = x_orig.shape[0]
    small_const = 1e-10
    x = x_orig.view(bs, -1).clamp(small_const, 1. - small_const)
    w = w_orig.view(bs, -1)
    gamma = x * (w < 0.) + (1. - x) * (w > 0.)
    delta = gamma.clone()

    w = w.abs()

    ind = gamma == 0.  # gamma < small_const #gamma == 0.
    # gamma *= (1 - ind.float())
    # delta *= (1 - ind.float())
    gamma_adj, w_adj = gamma.clone(), w.clone()
    gamma_adj[ind] = small_const
    w_adj[ind] = 0.

    mus = w_adj / (p * (gamma_adj ** (p - 1)))
    if verbose:
        print('mus nan in tensor', mus.isnan().any())
        # print('w is nan', w.isnan().any())
    mussorted, ind = mus.sort(dim=1)
    gammasorted, wsorted = gamma.gather(1, ind), w_adj.gather(1, ind)

    # print(mussorted[-1])
    # print(gammasorted.min()

    gammacum = torch.cat([torch.zeros([bs, 1], device=x.device),
                          (gammasorted ** p).cumsum(dim=1)],  # .fliplr()
                         # torch.zeros([bs, 1], device=x.device),
                         dim=1)
    gammacummax = gammacum.max(1)[0].unsqueeze(1)
    # print(gammacum.max().item(), gammacum.min().item())
    # gammacum = (gammasorted ** p).sum(dim=-1, keepdim=True) - gammacum
    gammacum = gammacum[:, -1].unsqueeze(1) - gammacum
    # print(gammacum.max().item(), gammacum.min().item())
    gammacum.clamp_(0.)  # small_const
    # gammacum = torch.min(gammacum, gammacummax * torch.ones_like(gammacum))
    # print(gammacum.max().item(), gammacum.min().item())

    wcum = (wsorted ** (p / (p - 1))).cumsum(dim=1)

    # print(gammacum[-1]) #wcum[-1]
    # mussorted[mussorted==0] = small_const
    # mussorted[mussorted.abs() < small_const] = small_const
    denominator = (p * mussorted) ** (p / (p - 1))
    denominator[denominator < small_const] = small_const
    mucum = torch.cat([torch.zeros([bs, 1], device=x.device),
                       wcum / denominator], dim=1)
    if verbose:
        print('mucum is nan', mucum.isnan().any())
        print('wcum is nan', wcum.isnan().any())
        print('wsorted is nan', wsorted.isnan().any())
        print('w_adj is nan', w_adj.isnan().any())
        print('w is nan', w.isnan().any())
    fs = gammacum + mucum - eps ** p
    # print(fs[-1], gammacum[-1], mucum[-1])

    ind = fs[:, 0] > 0.  # * (fs[-1] < 0.)
    # print(ind)
    lb = torch.zeros(bs).long()
    ub = lb + fs.shape[1]

    u = torch.arange(bs)
    for c in range(math.ceil(math.log2(fs.shape[1]))):
        a = (lb + ub) // 2
        indnew = fs[u, a] > 0.  # - 1e-6
        lb[indnew] = a[indnew].clone()
        ub[~indnew] = a[~indnew].clone()

    # print(lb, ub)
    # lb += 1
    # print(wcum.min(), (eps ** p - gammacum[u, lb]).min())
    # if (eps ** p - gammacum[u, lb]).min() < 0:
    #    print(f'found value={(eps ** p - gammacum[u, lb]).min().item()}')
    pmstar = wcum[u, lb - 1] / (eps ** p - gammacum[u, lb]).clip(small_const)  # wcum[u, lb]

    if verbose:
        print('pmstar is nan', pmstar.isnan().any())
        print('pmstar pow has nan', (pmstar ** (1 / p)).isnan().any())
        '''ind_test = (pmstar ** (1 / p)).view(-1) != (pmstar ** (1 / p)).view(-1)
        print(ind_test, 1 / p, '\n', pmstar.view(-1)[ind_test], '\n', (pmstar ** (1 / p)).view(-1)[ind_test])
        print(pmstar, pmstar.shape)
        print(pmstar ** (1 / p))'''
    # pmstar[pmstar == 0] = small_const
    # pmstar[(pmstar ** (1 / p)).abs() < small_const] = small_const
    deltamax = w ** (1 / (p - 1)) / pmstar.unsqueeze(1) ** (1 / p)  # ** (1 / (p - 1))
    if verbose:
        print('deltamax is nann', deltamax.isnan().any())
        # print((pmstar.unsqueeze(1).repeat([1, deltamax.shape[1]]) ** (1 / p))[deltamax != deltamax])
        # print(w[deltamax != deltamax])
    # print(deltamax)
    delta[ind] = torch.min(delta[ind],  # deltamax[ind].unsqueeze(1
                           # ) * torch.ones_like(delta[ind])
                           deltamax[ind])

    res = delta.view(bs, -1).norm(p=p, dim=1)[ind]
    # res_other = Lp_norm(delta, p)[~ind]
    print(res.max().item(), res.min().item())
    # print(f'max {p} pert={res.max():.5f}, {res_other.max():.5f}')

    # indpos, indneg = res > eps * 1.1, res < eps * .9

    print('\n')

    return delta.view(w_orig.shape) * w_orig.sign()

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


def project_perturbation(perturbation, eps, p, center=None):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        print('l2 renorm')
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        ##pert_normalized = project_onto_l1_ball(perturbation, eps)
        ##return pert_normalized
        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized
    #elif p in ['LPIPS']:
    #    pert_normalized = project_onto_LPIPS_ball(perturbation, eps)
    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')

class ActivationAPGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    :param last_layer     last layer of the PyTorch model
    :param eigenvecs:     eigenvectors of PCA on weighted activations
    :param target_class:  determines which weights are used for the weighted activations
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=True,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            ODI_steps=None,
            fw_momentum=0,
            return_all_losses_confid=True,
            pgd_mode=False,
            pgd_step_size=None,
            fw_constraint='intersection',
            masks=1,
            dist_regularizer=None,
            second_classifier=None,
            last_layer=None,
            eigenvecs=None,
            target_cls=0,
            reg_other=1.0,
            ica_components=None,
            ica_mean=None,
            minimize=False,
            minimize_abs=False
    ):
        """
        AutoPGD implementation in PyTorch
        """

        self.masks = masks
        print('masks used', type(self.masks), self.masks)

        self.dist_regularizer = dist_regularizer
        self.second_classifier = second_classifier
        self.ODI_steps = ODI_steps

        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho

        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        # self.init_point = None
        self.use_largereps = use_largereps
        # self.larger_epss = None
        # self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.fw_momentum = fw_momentum
        self.return_all_losses_confid = return_all_losses_confid
        self.pgd_mode = pgd_mode
        self.pgd_step_size = pgd_step_size
        self.fw_constraint = fw_constraint
        
        self.last_layer = last_layer
        self.eigenvecs = torch.tensor(eigenvecs, device=device, dtype=torch.float32)
        self.target_cls = target_cls
        self.reg_other = reg_other 
        self.loss_func = loss
        self.ica_components = ica_components
        self.ica_mean = ica_mean
        self.n_ica_in = ica_components.shape[1] if ica_components is not None else None
        self.minimize = minimize
        self.minimize_abs = minimize_abs
        #rnd_sd.set_rand_seed(self.seed)
        
        if loss == 'obj':
            if not self.minimize:
                def loss_obj(alpha, y):
                    alpha = alpha * torch.sign(torch.sum(self.eigenvecs, dim=0, keepdim=True))
                    return alpha[np.arange(len(y)),y] 
            else:
                if self.minimize_abs:
                    print("######## CORRECT LOSS #########")
                    def loss_obj(alpha, y):
                        alpha = torch.abs(alpha)
                        alpha *= -1 * torch.sign(torch.sum(self.eigenvecs, dim=0, keepdim=True))
                        return alpha[np.arange(len(y)),y] 
                else:
                    def loss_obj(alpha, y):
                        alpha = -alpha * torch.sign(torch.sum(self.eigenvecs, dim=0, keepdim=True))
                        return alpha[np.arange(len(y)),y] 
            self.criterion_indiv = loss_obj
        if loss == 'obj_full':
            def loss_obj_full(alpha, y):
                if self.ica_components is not None:
                    if self.ica_mean is None:
                        alpha = alpha * torch.sum(self.eigenvecs[:,:self.n_ica_in]@self.ica_components.T, dim=0, keepdim=True)
                    else:
                        alpha = alpha * torch.sum(self.ica_components.T, dim=0, keepdim=True)
                else:
                    alpha = alpha * torch.sum(self.eigenvecs, dim=0, keepdim=True)
                return alpha[np.arange(len(y)),y] 
            self.criterion_indiv = loss_obj_full
        elif loss == 'min_other':
            def loss_min_other(alpha, y):
                alpha_target = alpha[np.arange(len(y)), y]
                alpha_other = torch.sum(alpha ** 2, dim=1) - alpha_target ** 2
                n_other = alpha.shape[1] - 1
                return alpha_target - self.reg_other * alpha_other/n_other
            self.criterion_indiv = loss_min_other
        elif loss == 'min_other_eig':
            def loss_min_other(alpha, y):
                alpha = alpha * torch.sum(self.eigenvecs,dim=0)
                alpha_target = alpha[np.arange(len(y)), y]
                alpha_other = torch.sum(alpha ** 2, dim=1) - alpha_target ** 2
                n_other = alpha.shape[1] - 1
                return alpha_target - self.reg_other * alpha_other/n_other
            self.criterion_indiv = loss_min_other
        elif loss == 'max_comp_conf':
            def loss_max_comp_conf(alpha, y, activations):
                out = activations@self.last_layer.weight.T + self.last_layer.bias
                alpha = alpha * torch.sum(self.eigenvecs, axis=0)
                alpha_target = alpha[np.arange(len(y)), y]
                return alpha_target  - torch.log(torch.sum(torch.exp(out)))
            self.criterion_indiv = loss_max_comp_conf
        elif loss == 'ce':
            cross_entropy = nn.CrossEntropyLoss(reduction='none')
            def loss_ce(alpha, y):
                alpha_soft = alpha.softmax(1)
                return -1 * cross_entropy(alpha_soft, y)
            self.criterion_indiv = loss_ce
        elif loss == 'ce_abs':
            cross_entropy = nn.CrossEntropyLoss(reduction='none')
            def loss_ce_abs(alpha, y):
                alpha_abs = torch.abs(alpha)
                alpha_abs[np.arange(len(y)), y] = alpha[np.arange(len(y)), y] 
                alpha_soft = alpha_abs.softmax(1)
                return -1 * cross_entropy(alpha_soft, y)
            self.criterion_indiv = loss_ce_abs
        elif loss == 'log_nll':
            nll = nn.NLLLoss(reduction='none')
            def loss_log_nll(alpha, y):
                alpha_log = alpha.log()
                return -1 * nll(alpha_log, y)
            self.criterion_indiv = loss_log_nll
        elif loss == 'soft_log_nll':
            nll = nn.NLLLoss(reduction='none')
            def loss_soft_log_nll(alpha, y):
                alpha_log = alpha.softmax(1).log()
                return -1 * nll(alpha_log, y)
            self.criterion_indiv = loss_soft_log_nll
        print('using loss', loss)

    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        
        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
        """
        else:
            print('halved checkpoints params-')
            self.n_iter_2 = max(int(0.05 * self.n_iter), 1)
            self.n_iter_min = max(int(0.04 * self.n_iter), 1)
            self.size_decr = min(-int(0.2 * self.n_iter), -1)
        """

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
                1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def get_init(self, x):

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                                                   ).detach() * self.normalize(t) * self.masks
        elif self.norm == 'L2':

            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                                                   ).detach() * self.normalize(t) * self.masks
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + (t + delta)*self.masks
    
        return x_adv-x

    def attack_single_run(self, x, y, x_init=None, first_run=False):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        all_losses = []

        if x_init is not None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        elif first_run:
            print('First run, intialized with an image.')
            x_adv = x.clone()
        else:
            print('Standard normalization is being used norm is', self.norm)
            if self.norm == 'Linf':
                t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
                x_adv = x + self.eps * torch.ones_like(x
                                                       ).detach() * self.normalize(t) * self.masks
            elif self.norm == 'L2':
                t = torch.randn(x.shape).to(self.device).detach()
                x_adv = x + self.eps * torch.ones_like(x
                                                       ).detach() * self.normalize(t) * self.masks
            elif self.norm == 'L1':
                t = torch.randn(x.shape).to(self.device).detach()
                delta = L1_projection(x, t, self.eps)
                x_adv = x + (t + delta) * self.masks

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
                                 ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
                                      ).to(self.device)
        assert not self.is_tf_model
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        if not self.pgd_mode:
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                    weighted_act = activations * self.last_layer.weight[self.target_cls]

                    pca_activations = weighted_act@self.eigenvecs
                    if self.ica_components is not None:
                        if self.ica_mean is not None:
                            pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                        else:
                            pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
                    
                    if self.loss_func == 'max_comp_conf':
                        loss_indiv = self.criterion_indiv(pca_activations, y, activations)
                    else:
                        loss_indiv = self.criterion_indiv(pca_activations, y)

                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                
            grad /= float(self.eot_iter)
        else:
            with torch.enable_grad():
                activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                weighted_act = activations * self.last_layer.weight[self.target_cls]
                
                pca_activations = weighted_act@self.eigenvecs
                if self.ica_components is not None:
                    if self.ica_mean is not None:
                        pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                    else:
                        pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
                if self.loss_func == 'max_comp_conf':
                    loss_indiv = self.criterion_indiv(pca_activations, y, activations)
                else:
                    loss_indiv = self.criterion_indiv(pca_activations, y)

                loss = loss_indiv.sum()
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            print('pgd mode is used with stepsize', self.pgd_step_size)
        grad_best = grad.clone()

        loss_best = loss_indiv.detach().clone()
        if self.return_all_losses_confid:
            #activations = self.ll_activations.attribute(x, attribute_to_layer_input=True)
            activations = activations_with_grad(x, self.model, self.last_layer)
            weighted_act = activations * self.last_layer.weight[self.target_cls]

            pca_activations = weighted_act@self.eigenvecs 
            if self.ica_components is not None:
                if self.ica_mean is not None:
                    pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                else:
                    pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
            if self.loss_func == 'max_comp_conf':
                loss_indiv_ = self.criterion_indiv(pca_activations, y, activations)
            else:
                loss_indiv_ = self.criterion_indiv(pca_activations, y)
            
            all_losses.append(loss_indiv_.detach().cpu().unsqueeze(dim=1))
            print('all losses', all_losses)

        print('loss indiv shape is', loss_indiv.shape)

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2

        if self.pgd_mode:
            step_size = self.pgd_step_size
        else:
            step_size = alpha * self.eps * torch.ones([x.shape[0], *(
                [1] * self.ndims)]).to(self.device).detach()

        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old = n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            adasp_redstep = 1.5
            adasp_minstep = 10.
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]
        u = torch.arange(x.shape[0], device=self.device)
        for i in tqdm(range(self.n_iter)):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * self.masks * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                                                              x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.masks * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                                                             ) * torch.min(self.eps * torch.ones_like(x).detach(),
                                                                           self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                                                             ) * torch.min(self.eps * torch.ones_like(x).detach(),
                                                                           self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * self.masks * sparsegrad.sign() / (
                            sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                                -1, *[1] * (len(x.shape) - 1)) + 1e-10)

                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            if not self.pgd_mode:
                for _ in range(self.eot_iter):
                    with torch.enable_grad():
                        activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                        weighted_act = activations * self.last_layer.weight[self.target_cls]

                        pca_activations = weighted_act@self.eigenvecs
                        if self.ica_components is not None:
                            if self.ica_mean is not None:
                                pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                            else:
                                pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
                        if self.loss_func == 'max_comp_conf':
                            loss_indiv = self.criterion_indiv(pca_activations, y, activations)
                        else:
                            loss_indiv = self.criterion_indiv(pca_activations, y)

                        loss = loss_indiv.sum()

                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                grad /= float(self.eot_iter)
            else:
                with torch.enable_grad():
                    activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                    weighted_act = activations * self.last_layer.weight[self.target_cls]
                    
                    pca_activations = weighted_act@self.eigenvecs
                    if self.ica_components is not None:
                        if self.ica_mean is not None:
                            pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                        else:
                            pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
                    if self.loss_func == 'max_comp_conf':
                        loss_indiv = self.criterion_indiv(pca_activations, y, activations)
                    else:
                        loss_indiv = self.criterion_indiv(pca_activations, y)

                    loss = loss_indiv.sum()
                    grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} {}'.format(
                    i, loss_best.sum(),  str_stats))

            if not self.pgd_mode:
                ### check step size
                with torch.no_grad():
                    y1 = loss_indiv.detach().clone()
 
                    loss_steps[i] = y1 + 0
                    ind = (y1 > loss_best).nonzero().squeeze()
                    x_best[ind] = x_adv[ind].clone()
                    grad_best[ind] = grad[ind].clone()
                    
                    loss_best[ind] = y1[ind] + 0
                    loss_best_steps[i + 1] = loss_best + 0

                    counter3 += 1

                    if counter3 == k:
                        if self.norm in ['Linf', 'L2']:

                            fl_oscillation = self.check_oscillation(loss_steps, i, k,
                                                                    loss_best, k3=self.thr_decr)
                            fl_reduce_no_impr = (1. - reduced_last_check) * (
                                    loss_best_last_check >= loss_best).float()
                            fl_oscillation = torch.max(fl_oscillation,
                                                       fl_reduce_no_impr)
                            reduced_last_check = fl_oscillation.clone()
                            loss_best_last_check = loss_best.clone()

                            if fl_oscillation.sum() > 0:
                                ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                                step_size[ind_fl_osc] /= 2

                                x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                                grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                            print('prev k is', k)
                            k = max(k - self.size_decr, self.n_iter_min)
                            print('next k is', k)

                        elif self.norm == 'L1':
                            sp_curr = L0_norm(x_best - x)
                            fl_redtopk = (sp_curr / sp_old) < .95
                            topk = sp_curr / n_fts / 1.5
                            step_size[fl_redtopk] = alpha * self.eps
                            step_size[~fl_redtopk] /= adasp_redstep
                            step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                            sp_old = sp_curr.clone()

                            x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                            grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                        counter3 = 0

            if self.return_all_losses_confid:
                activations = activations_with_grad(x_adv, self.model, self.last_layer)
                weighted_act = activations * self.last_layer.weight[self.target_cls]
                
                pca_activations = weighted_act@self.eigenvecs
                if self.ica_components is not None:
                    if self.ica_mean is not None:
                        pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
                    else:
                            pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
                if self.loss_func == 'max_comp_conf':
                    loss_indiv_ = self.criterion_indiv(pca_activations, y, activations)
                else:
                    loss_indiv_ = self.criterion_indiv(pca_activations, y)
                
                all_losses.append(loss_indiv_.cpu().unsqueeze(dim=1))
                if self.verbose:
                    print('losses:', all_losses[-1])
                    print('pca_activations:', pca_activations)

        if self.return_all_losses_confid:
            return (x_best, loss_best, x_best_adv, torch.cat(all_losses, dim=1))
        else:
            return (x_best, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """              
        activations = activations_with_grad(x, self.model, self.last_layer)
        weighted_act = activations * self.last_layer.weight[self.target_cls]

        pca_activations = weighted_act@self.eigenvecs
        if self.ica_components is not None:
            if self.ica_mean is not None:
                pca_activations = (weighted_act-self.ica_mean)@self.ica_components.T
            else:
                pca_activations = pca_activations[:, :self.n_ica_in]@self.ica_components.T
        if self.loss_func == 'max_comp_conf':
            loss_indiv_ = self.criterion_indiv(pca_activations, y, activations)
        else:
            loss_indiv_ = self.criterion_indiv(pca_activations, y)

        print('start loss 2', loss_indiv_)

        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        
        assert not self.is_tf_model
    
        y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.verbose:
            print('-------------------------- ',
                  'running {}-attack with epsilon {:.5f}'.format(
                      self.norm, self.eps),
                  '--------------------------')
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                     .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])  # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                                                                    ) for c in epss]),
                                                      '+'.join([str(c) for c in iters])))

        startt = time.time()
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
        for counter in range(self.n_restarts):

            if not self.use_largereps:
                res_curr = self.attack_single_run(x, y, x_init=x_init)  # , first_run=(counter==0))
            else:
                res_curr = self.decr_eps_pgd(x, y, epss, iters)

            if self.return_all_losses_confid:
                print('all confids and losses are returned, restart num', counter)
                best_curr, loss_curr, adv_curr, all_losses = res_curr
            else:
                best_curr, loss_curr, _ = res_curr  # , first_run=(counter==0))
            ind_curr = (loss_curr > loss_best).nonzero().squeeze()
            adv_best[ind_curr] = best_curr[ind_curr] + 0.

            loss_best[ind_curr] = loss_curr[ind_curr] + 0.

            if self.verbose:
                print('restart {} - loss: {:.5f}'.format(
                    counter, loss_best.sum()))
        if self.return_all_losses_confid:
            return adv_best, all_losses
        else:
            return adv_best

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])
        mask = x_sorted[u, -1] == x[u, y]

        return -(torch.where(mask, x_sorted[u, -2], x_sorted[u, -1]) - x[u, y]) / (x_sorted[:, -1] - .5 * (
                x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def margin_targeted(self, x, y):
        u = torch.arange(x.shape[0])
        return -(x[u, y] - x[u, self.y_target])

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):

        assert len(epss) == len(iters)
        # ToDo - add other numeric norms for fw
        assert self.norm == 'L1'
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            if self.norm == 'L1':
                x_init = x + torch.randn_like(x)
                x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
            if self.verbose:
                print('x_init-x norms is', (x_init - x).view(x.shape[0], -1).norm(p=get_norm(self.norm), dim=1))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if x_init is not None:
                if self.norm == 'L1':
                    x_init += L1_projection(x, x_init - x, 1. * eps)
                if self.verbose:
                    print('x_init-x norms is', (x_init - x).view(x.shape[0], -1).norm(p=get_norm(self.norm), dim=1))
            res = self.attack_single_run(x, y, x_init=x_init)
            x_init = res[0]

        return res