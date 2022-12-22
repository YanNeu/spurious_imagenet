import torch

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = torch.tensor(mean)
        std = torch.tensor(std)

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x, *args, **kwargs):
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()

def IdentityWrapper(model):
    mean = [0., 0., 0.]
    std = [1., 1., 1.]
    return NormalizationWrapper(model, mean, std)

def Cifar10Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def Cifar100Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def SVHNWrapper(model):
    mean = torch.tensor([0.4377, 0.4438, 0.4728])
    std = torch.tensor([0.1201, 0.1231, 0.1052])
    return NormalizationWrapper(model, mean, std)

def CelebAWrapper(model):
    mean = torch.tensor([0.5063, 0.4258, 0.3832])
    std = torch.tensor([0.2632, 0.2424, 0.2385])
    return NormalizationWrapper(model, mean, std)

def TinyImageNetWrapper(model):
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2302, 0.2265, 0.2262])
    return NormalizationWrapper(model, mean, std)

def ImageNetWrapper(model):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return NormalizationWrapper(model, mean, std)

def RestrictedImageNetWrapper(model):
    mean = torch.tensor([0.4717, 0.4499, 0.3837])
    std = torch.tensor([0.2600, 0.2516, 0.2575])
    return NormalizationWrapper(model, mean, std)

def BigTransferWrapper(model):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    return NormalizationWrapper(model, mean, std)

def LSUNScenesWrapper(model):
    #imagenet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return NormalizationWrapper(model, mean, std)
