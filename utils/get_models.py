import torch
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.load_trained_model import load_model
from utils.salient_imagenet_model import load_robust_model
from utils.model_normalization import NormalizationWrapper
from dataset.spurious_projection_wrapper import wrap_model


def get_model(device, device_ids, model_name):
    if model_name == 'spurious_projection_robust_resnet':
        model, input_size = get_model(device, None, 'robust_resnet')
        model = wrap_model(model, device)
    elif model_name == 'robust_resnet':
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