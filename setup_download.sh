#!/bin/bash

# Download Spurious ImageNet
cd dataset/spurious_imagenet
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/xBtTG9jnACJDxSg/download/images_100_classes.zip
unzip images_100_classes.zip
rm images_100_classes.zip
cd ../..

# Download precomputed alpha values (for SpuFix)
cd neural_pca
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/4sfzFZcF3TKsWJ2/download/spurious_alphas_train.zip
unzip spurious_alpha_train.zip
rm spurious_alpha_train.zip
cd ..
