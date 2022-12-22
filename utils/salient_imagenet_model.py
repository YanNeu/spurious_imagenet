from robustness import model_utils
from robustness import datasets as dataset_utils
import torch

from utils.datasets.imagenet import get_imagenet_path

def load_robust_model():
    dataset_function = getattr(dataset_utils, 'ImageNet')
    dataset = dataset_function(get_imagenet_path())

    class MadryWrapper(torch.nn.Module):
        def __init__(self, model, normalizer):
            super().__init__()
            self.model = model
            self.normalizer = normalizer

        def forward(self, img):
            normalized_inp = self.normalizer(img)
            output = self.model(normalized_inp, with_latent=False,
                                fake_relu=False, no_relu=False)
            return output

    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': '../utils/ImageNetModels/imagenet_l2_3_0.pt',
        'parallel': False
    }
    
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model = MadryWrapper(model.model, model.normalizer)
    model.eval()
    return model