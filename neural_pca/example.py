import sys
sys.path.insert(0,'../.')

import torch
from torchvision.transforms import PILToTensor
from PIL import Image

from utils.load_trained_model import load_model
from utils.temperature_wrapper import TemperatureWrapper

from neural_pca.counterfactual import create_results_dir
from neural_pca.activation_space import ActivationSpace
from neural_pca.data import imagenet_label2class, imagenet_subset
from neural_pca.visualization import visualize_components

device = torch.device('cuda:0')

model_description = ('PytorchResNet50', 'l2_improved_3_ep', 'best', 0.7155761122703552, False)
type, model_folder, model_checkpoint, temperature, temp = model_description
model = load_model(type, model_folder, model_checkpoint, temperature, device, load_temp=temp, dataset="imagenet")

# Model
model = TemperatureWrapper(model, temperature)
last_layer = model.model.model.fc

# Load Activation Space
target_class = 94 # hummingbird
component_idx = 1 # hummingbird feeder
load = False
batchsize = 32
n_workers = 8

save_dir = create_results_dir(f"act_spaces/{target_class}_{imagenet_label2class[target_class]}")
if load:
    act_space = ActivationSpace(model, last_layer, target_class, device)
    act_space.load(f'act_spaces/{target_class}_{imagenet_label2class[target_class]}/act_space.npy')
else:
    train_loader, _ = imagenet_subset(target_class, batch_size=batchsize, n_workers=n_workers)
    act_space = ActivationSpace(model, last_layer, target_class, device, train_loader=train_loader)
    act_space.fit()
    act_space.save(f'act_spaces/{target_class}_{imagenet_label2class[target_class]}/act_space.npy')

# Compute alpha for new image
img = PILToTensor()(Image.open('../example_images/example_img.jpg'))/255

alphas = act_space.transform(img)

print(f'alpha {alphas[component_idx].item()}')
print(f'Max value in training set {torch.max(act_space.pca_train[:, component_idx]).item()}')

# Visualize top 10 components of this class
visualize_components(target_class, device, batchsize=batchsize)