# Spurious Features Everywhere - Large-Scale Detection of Harmful Spurious Features in ImageNet
Yannic Neuhaus, Maximilian Augustin, Valentyn Boreiko, Matthias Hein\
*University of Tübingen*

Accepted to ICCV 2023

This repository will contain the code for our paper [Spurious Features Everywhere - Large-Scale Detection of Harmful Spurious Features in ImageNet](https://arxiv.org/abs/2212.04871) including the *Spurious ImageNet* dataset.

In this paper, we develop a framework that allows us to systematically identify spurious features in large datasets like ImageNet. It is based on our neural PCA components and their visualization.
By applying this framework (including minimal human supervision) to ImageNet, we identified 319 neural PCA components corresponding to spurious features of 230 ImageNet classes. For 100 of these features, we validated our results by 
collecting images from the OpenImages dataset which show the spurious feature and do not contain the actual class object but are still classified as this class. 

<p align="center">
  <img width="505" src="./example_images/teaser.png">
</p>

## The *Spurious ImageNet* Dataset
We selected 100 of our spurious features and collected 75 images from the top-ranked images in OpenImages according to the value of $\alpha_l^{(k)}$, each containing only the spurious feature but not the class object.
This dataset can be used to measure the reliance of image classifiers on spurious features. It has the advantage that it consists of real images and thus provides a realistic impression of the performance of ImageNet classifiers in 
the wild. 

### Examples
<p align="center">
  <img width="1390" src="./example_images/spur_in_overview_0.png">
</p>
<p align="center">
  <img width="1390" src="./example_images/spur_in_overview_1.png">
</p>
<p align="center">
  <img width="1390" src="./example_images/spur_in_overview_2.png">
</p>
<p align="center">
  <img width="1390" src="./example_images/spur_in_overview_3.png">
</p>

### Setup
#### Download Images
Clone this repository and download the dataset:
```
git clone git@github.com:YanNeu/spurious_imagenet.git
cd spurious_imagenet/dataset/spurious_imagenet
wget https://www.dropbox.com/s/bhdi7iz4rtmn0ud/images_100_classes.zip
unzip images_100_classes.zip
rm images_100_classes.zip 
```

#### Adjust `imagenet_path`
Open `utils/datasets/paths.py` and adjust the `base_data_path` in line 6, the default value is `/scratch/datasets/`. Note that we assume that you have extracted ILSVRC2012 to `base_data_path/imagenet`. If this does not match your folder layout, you can also directly manipulate `get_imagenet_path` in line 64. For example if your dataset is located in `/home/user/datasets/ilsvrc2012/` you could change the function to:

```
def get_imagenet_path():  
    path = `/home/user/datasets/ilsvrc2012/` 
    return path
```

#### Robust ResNet50
Download the weights from [here](https://drive.google.com/file/d/169fhxn5X2_1-5vWTepkKJZRMdr8z4b9p/view?usp=sharing) into `utils` and unzip the model.

### Required Packages
Required packages are listed in `reqs.yml`. The `robustness` package needs to be installed from the corresponding github repo:
```
conda env create -f reqs.yml
conda activate spurious_imagenet
cd utils
wget https://github.com/MadryLab/robustness/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
pip install -e robustness-master
```

❗Models from timm updates after Oct 10, 2022 require 0.8.x pre-releases (`pip install --pre timm`) or cloning the main branch of [its](https://github.com/huggingface/pytorch-image-models) GitHub repository❗


### Compute Spurious mAUC
A classifier $f$ not relying on the spurious feature should predict a low probability for class $k$ for the Spurious ImageNet samples, especially compared to test set images of ImageNet for class $k$. Thus, for each class, we measure the AUC (area under the curve) for the separation of images with the spurious features but not showing class $k$ versus test set images of class $k$ according to the predicted probability for class $k$. A classifier not depending on the spurious feature should easily attain a perfect AUC of 1. We compute the mean AUC over all 40 classes.

Use `dataset/spurious_score.py` and replace `get_model` to evaluate your model. A table with results will be saved as `dataset/spurious_imagenet/evaluation/<*model name*>/spurious_score.txt`:

```
model_name = "robust_resnet"
model, img_size = get_model(device, device_ids, model_name)
    
eval_auc(model, model_name, img_size, device, bs)
```


## Class-wise Neural PCA
The folder `neural_pca` contains all code to compute the class-wise neural PCA components of ImageNet classes and corresponding visualisations. The script `neural_pca/example.py` shows how to compute the $\alpha$ values and visualisations for the top 10 components for a given class.

## Citation

```bibtex
@article{neuhaus2022spurious,
  title={Spurious Features Everywhere--Large-Scale Detection of Harmful Spurious Features in ImageNet},
  author={Neuhaus, Yannic and Augustin, Maximilian and Boreiko, Valentyn and Hein, Matthias},
  journal={arXiv preprint arXiv:2212.04871},
  year={2022}
}
```


