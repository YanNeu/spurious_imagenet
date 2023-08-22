import sys
sys.path.insert(0,'..')

import os

import numpy as np
import torch

import argparse
import pathlib
from sklearn.metrics import average_precision_score

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.datasets import ImageNet

from utils.model_normalization import NormalizationWrapper
from utils.datasets.imagenet import get_imagenet_path, get_imagenet_labels
from utils.load_trained_model import load_model
from utils.salient_imagenet_model import load_robust_model
from spurious_dataset import get_spurious_datasets, get_imagenet_matching_subset

from prettytable import PrettyTable


spurious_imagenet_dir = 'spurious_imagenet'
dataset_dir = os.path.join(spurious_imagenet_dir, 'images')
result_dir = os.path.join(spurious_imagenet_dir, 'evaluation')

in_labels = get_imagenet_labels()


def get_google_in1k_to_21k_map():
    in_path = get_imagenet_path()
    imagenet = ImageNet(in_path, split='val', transform=None)

    with open('../utils/imagenet21k_wordnet_ids.txt') as f:
        in21k_wids = [wid.rstrip() for wid in f.readlines()]

    idx_map = {}
    for wid, idx in imagenet.wnid_to_idx.items():
        try:
            idx_map[idx] = in21k_wids.index(wid)
        except ValueError:
            print(f'No match in in21k: {idx} - {wid} - {in_labels[idx]}')

    return idx_map

google_in_1k_to_21k = get_google_in1k_to_21k_map()
google_in_21k_to_1k = {v: k for (k, v) in google_in_1k_to_21k.items()}

def get_in1k_to_21k_map():
    in_path = get_imagenet_path()
    imagenet = ImageNet(in_path, split='val', transform=None)

    with open('../utils/imagenet21_wordnet_ids_non_google.txt') as f:
        in21k_wids = [wid.rstrip() for wid in f.readlines()]

    idx_map = {}
    for wid, idx in imagenet.wnid_to_idx.items():
        try:
            idx_map[idx] = in21k_wids.index(wid)
        except ValueError:
            print(f'No match in in21k: {idx} - {wid} - {in_labels[idx]}')

    return idx_map

in_1k_to_21k = get_in1k_to_21k_map()
in_21k_to_1k = {v: k for (k, v) in in_1k_to_21k.items()}


def get_probabilities_targets(loader, model, device):
    dataset_length = len(loader.dataset)
    targets = torch.zeros(dataset_length, dtype=torch.long)
    probabilities = None

    idx = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            idx_next = idx + len(data)
            out = model(data)
            probs = torch.softmax(out, dim=1)
            targets[idx:idx_next] = target

            if idx == 0:
                #as we use in22k classifiers, output size can't be determined easily
                probabilities = torch.zeros((dataset_length, out.shape[1]))

            probabilities[idx:idx_next] = probs.detach().cpu()
            idx = idx_next

    return probabilities, targets

def map_index(num_clases, class_idx):
    if num_clases == 1000:
        mapped_idx = class_idx
    elif num_clases == 21843:
        #vit/bit
        mapped_idx = google_in_1k_to_21k[class_idx]
    elif num_clases == 21841:
        #convnext/beit
        mapped_idx = in_1k_to_21k[class_idx]
    elif num_clases == 10450:
        #In-21k-P == Winter2021 Release
        #https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return mapped_idx

def calculate_score_frac_classified_as(probabilities, class_idx, gt_targets):
    idcs_matching = gt_targets == class_idx
    mapped_idx = map_index(probabilities.shape[1], class_idx)
    scores = probabilities[idcs_matching, mapped_idx].numpy()
    _, preds = torch.max(probabilities[idcs_matching], dim=1)
    frac_classified_as = torch.mean((preds == mapped_idx).float()).item()
    return scores, frac_classified_as


def calculate_confs_on_correct(probabilities, class_idx, gt_targets):
    gt_matching_idcs = torch.nonzero(gt_targets == class_idx, as_tuple=False).squeeze()
    mapped_idx = map_index(probabilities.shape[1], class_idx)
    _, preds = torch.max(probabilities[gt_matching_idcs], dim=1)
    pred_gt_matching_idcs = gt_matching_idcs[preds == mapped_idx]

    confs = probabilities[pred_gt_matching_idcs, mapped_idx].numpy()
    frac_classified_as = torch.mean((preds == mapped_idx).float()).item()
    return confs, frac_classified_as


def calculate_spurious_score(in_loader, out_loader, device, model):
    in_probs, in_targets = get_probabilities_targets(in_loader, model, device)
    out_probs, out_targets = get_probabilities_targets(out_loader, model, device)

    classes = set(out_loader.dataset.targets)

    class_idcs = np.zeros(len(classes), dtype=int)
    class_auc_scores = np.zeros(len(classes), dtype=float)

    class_in_correct_classified_as = np.zeros(len(classes), dtype=float)
    class_spurious_wrongly_classified_as = np.zeros(len(classes), dtype=float)

    class_in_probability_into = np.zeros(len(classes), dtype=float)
    class_spurious_probability_into = np.zeros(len(classes), dtype=float)


    in_scores_merged = np.zeros(len(in_probs))
    out_scores_merged = np.zeros(len(out_probs))

    in_idx = 0
    out_idx = 0
    #AUCs
    for i, class_idx in enumerate(classes):
        in_scores, in_frac_corrrect_classified_as = calculate_score_frac_classified_as(in_probs, class_idx, in_targets)

        in_scores_merged[in_idx:(in_idx+len(in_scores))] = in_scores
        in_idx += len(in_scores)

        out_scores, spurious_frac_wrongly_classified_as = calculate_score_frac_classified_as(out_probs, class_idx, out_targets)
        out_scores_merged[out_idx:(out_idx+len(out_scores))] = out_scores
        out_idx += len(out_scores)

        y_true = len(in_scores) * [1] + len(out_scores) * [0]
        y_score = np.concatenate([in_scores, out_scores])
        class_score = average_precision_score(y_true, y_score)
        class_auc_scores[i] = class_score
        class_idcs[i] = class_idx
        class_in_correct_classified_as[i] = in_frac_corrrect_classified_as
        class_spurious_wrongly_classified_as[i] = spurious_frac_wrongly_classified_as
        class_in_probability_into[i] = np.mean(in_scores)
        class_spurious_probability_into[i] = np.mean(out_scores)

    mean_auc_score = np.mean(class_auc_scores)
    y_true = len(in_scores_merged) * [1] + len(out_scores_merged) * [0]
    y_score = np.concatenate([in_scores_merged, out_scores_merged])
    joint_auc_score = average_precision_score(y_true, y_score)

    return mean_auc_score, joint_auc_score, class_auc_scores, \
           class_in_probability_into, class_spurious_probability_into,\
           class_in_correct_classified_as, class_spurious_wrongly_classified_as,\
           class_idcs


def get_model(device, device_ids, model_name):
    if model_name == 'robust_resnet':
        model_description = ('PytorchResNet50', 'l2_improved_3_ep', 'best', 0.7155761122703552, False)
        type, model_folder, model_checkpoint, temperature, temp = model_description
        model = load_model(type, model_folder, model_checkpoint, temperature, device, load_temp=temp,
                           dataset="imagenet")
        input_size = (3, 224, 224)
    elif model_name == 'resnet_salient_imagenet':
        model = load_robust_model()
        model = model.to(device)
        input_size = (3, 224, 224)
    else:
        model = timm.create_model(model_name, pretrained=True)

        try:
            default_cfg = model.default_cfg
            if 'test_input_size' in default_cfg:
                input_size = default_cfg['test_input_size']
            elif 'input_size' in default_cfg:
                input_size = default_cfg['input_size']

            mean = default_cfg['mean']
            std = default_cfg['std']
        except:
            print('Model default config could not be read')
            input_size = (3, 224, 224)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD

        model = NormalizationWrapper(model, mean=mean, std=std)
        model.eval()
        model.to(device)

        if device_ids is not None and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            print('Using DataParallel')

    return model, input_size

def calculate_accuracy(loader, model, device):
    with torch.no_grad():
        correct = 0
        N =0
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            out = model(data)
            conf, pred = torch.max( torch.softmax(out, dim=1), dim=1)
            correct += torch.sum(pred == target).item()
            N += len(data)

        acc = correct / N
        print(f'Accuracy {acc:.3f}')
        return acc

def get_loaders(img_size, bs):
    spurious_loader = get_spurious_datasets(dataset_dir, img_size=img_size[1], bs=bs, num_workers=8)
    included_classes = spurious_loader.dataset.included_classes
    in_subset_loader = get_imagenet_matching_subset(included_classes,
                                                        img_size=img_size[1], bs=bs, num_workers=8)
    return spurious_loader, in_subset_loader

def eval_spurious_score(model, model_name, device, spurious_loader, in_subset_loader):
    with open(os.path.join(dataset_dir, 'timestamp.txt'), 'r') as f:
        line = f.readline()
        dataset_timestamp = int(line.rstrip())

    model_results_dir = os.path.join(result_dir, model_name)
    pathlib.Path(model_results_dir).mkdir(parents=True, exist_ok=True)
    info_package_file = os.path.join(model_results_dir, 'info.pth')

    mean_auc_score, joint_auc_score, model_class_auc_scores, \
    class_in_probability_into, class_spurious_probability_into, \
    class_in_correct_classified_as, class_spurious_wrongly_classified_as, \
    class_idcs\
        = calculate_spurious_score(in_subset_loader, spurious_loader, device, model)

    info_package = {
        'mean_auc_score': mean_auc_score,
        'joint_auc_score': joint_auc_score,
        'class_auc_scores': model_class_auc_scores,
        'class_in_probability_into': class_in_probability_into,
        'class_spurious_probability_into': class_spurious_probability_into,
        'class_in_correct_classified_as': class_in_correct_classified_as,
        'class_spurious_wrongly_classified_as': class_spurious_wrongly_classified_as,
        'class_idcs': class_idcs,
        'timestamp': dataset_timestamp,
    }
    torch.save(info_package, info_package_file)

    #write txt
    out_file = os.path.join(model_results_dir, 'spurious_score.txt')
    with open(out_file, 'w') as f:
        f.write(f'Mean SpuriousScore: {mean_auc_score:.3f} - Joint SpuriousScore: {joint_auc_score:.3f}\n')
                
        pt = PrettyTable()
        pt.field_names = ['Idx',  'Class', 'Spurious AUC', "Validation Mean Prob", "Spurious Mean Prob",
                            'Frac Validation labeled as', 'Frac Spurious labeled as']
        pt.float_format = '.4'
        for class_idx, class_score, in_prob, spurious_prob, in_frac, spurious_frac\
                in zip(class_idcs, model_class_auc_scores,
                        class_in_probability_into, class_spurious_probability_into,
                        class_in_correct_classified_as, class_spurious_wrongly_classified_as):

                        pt.add_row([class_idx, in_labels[class_idx], class_score, in_prob, spurious_prob,
                                   in_frac, spurious_frac])

        f.write(pt.get_string())


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--bs', default=16, type=int,
                    help='batch size.')
    return parser.parse_args()


if __name__ == '__main__':
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

    bs = args.bs

    # Model
    model_name = "robust_resnet"
    model, img_size = get_model(device, device_ids, model_name)
    
    spurious_loader, in_subset_loader = get_loaders(img_size, bs)
    eval_spurious_score(model, model_name, device, spurious_loader, in_subset_loader)

