#!/bin/bash

# Download Spurious ImageNet
cd dataset/spurious_imagenet
wget 
unzip images_100_classes.zip
rm images_100_classes.zip
cd ../..

# Download precomputed alpha values (for SpuFix)
cd neural_pca
wget 
unzip spurious_alpha_train.zip
rm spurious_alpha_train.zip
cd ..
