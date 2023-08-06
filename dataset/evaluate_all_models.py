import os
import pathlib
import torch
from tqdm import tqdm
import argparse

from spurious_score import eval_spurious_score, get_model, get_loaders
from model_names import model_names


spurious_imagenet_dir = 'spurious_imagenet'
dataset_dir = os.path.join(spurious_imagenet_dir, 'images')
result_dir = os.path.join(spurious_imagenet_dir, 'evaluation')


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
    parser.add_argument('--bs', default=32, type=int,
                    help='batch size.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device, device_ids = get_devices(args.gpu)
    
    bs = args.bs

    with open(os.path.join(dataset_dir, 'timestamp.txt'), 'r') as f:
        line = f.readline()
        dataset_timestamp = int(line.rstrip())
    pbar = tqdm(model_names)
    for i, model_name in enumerate(pbar):

        model_results_dir = os.path.join(result_dir, model_name)
        pathlib.Path(model_results_dir).mkdir(parents=True, exist_ok=True)
        info_package_file = os.path.join(model_results_dir, 'info.pth')
        load_info_success = False
        if os.path.isfile(info_package_file):
            info_package = torch.load(info_package_file)
            if info_package['timestamp'] == dataset_timestamp:
                try:
                    mean_auc_score = info_package['mean_auc_score']
                    joint_auc_score = info_package['joint_auc_score']
                    model_class_auc_scores = info_package['class_auc_scores']
                    class_in_probability_into = info_package['class_in_probability_into']
                    class_spurious_probability_into = info_package['class_spurious_probability_into']
                    class_in_correct_classified_as = info_package['class_in_correct_classified_as']
                    class_spurious_wrongly_classified_as = info_package['class_spurious_wrongly_classified_as']
                    class_idcs = info_package['class_idcs']
                    load_info_success = True
                except KeyError:
                    pass
        if load_info_success: continue 

        pbar.set_description(f"{i}/{len(model_names)} {model_name}")
        model, img_size = get_model(device, device_ids, model_name)

        # load datasets
        spurious_loader, in_subset_loader = get_loaders(img_size, bs)
        eval_spurious_score(model, model_name, device, spurious_loader, in_subset_loader)
