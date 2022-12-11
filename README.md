# Spurious Features Everywhere - Large-Scale Detection of Harmful Spurious Features in ImageNet

This repository will contain the code for our paper [Spurious Features Everywhere - Large-Scale Detection of Harmful Spurious Features in ImageNet] including the *Spurious ImageNet* dataset.

In this paper, we develop a framework that allows us to systematically identify spurious features in large datasets like ImageNet. It is based on our neural PCA components and their visualization.
By applying this framework (including minimal human supervision) to ImageNet, we identified 319 neural PCA components corresponding to spurious features of 230 ImageNet classes. For 40 of these features, we validated our results by 
collecting images from the OpenImages dataset which show the spurious feature and do not contain the actual class object but are still classified as this class. 

<p align="center">
  <img width="505" height="531" src="./example_images/teaser.png">
</p>

## The *Spurious ImageNet* Dataset
We selected 40 of our spurious features and collected 75 images from the top-ranked images in OpenImages according to the value of $\alpha_l^{(k)}$.
<p align="center">
  <img width="937" height="1145" src="./example_images/examples_spurious_imagenet.jpg">
</p>
