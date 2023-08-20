import torch
import argparse

from spurious_score import eval_spurious_score, get_model, get_loaders

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
    parser.add_argument('--bs', default=16, type=int,
                    help='batch size.')
    parser.add_argument('--model', type=str, default='robust_resnet')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device, device_ids = get_devices(args.gpu)
    
    bs = args.bs

    # Model
    """
    evaluate pre-trained models from timm :
        just set the model_name accordingly
    
    evaluate your own models:
        - replace the get_model function
        - model should include a normalization wrapper (see utils.model_normalization.py)
        - img_size format (3, <size>, <size>)
    """

    model_name = args.model
    model, img_size = get_model(device, device_ids, model_name)
    
    # load datasets
    spurious_loader, in_subset_loader = get_loaders(img_size, bs)

    eval_spurious_score(model, model_name, device, spurious_loader, in_subset_loader)

