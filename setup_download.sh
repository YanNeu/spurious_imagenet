#!/bin/bash

# Download Spurious ImageNet
cd dataset/spurious_imagenet
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/xBtTG9jnACJDxSg/download/images_100_classes.zip
unzip images_100_classes.zip
rm images_100_classes.zip
cd ../..

# Download precomputed alpha values (for SpuFix)
cd neural_pca
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/FqbdADBoem3exgM/download/spurious_alphas_train.zip
unzip spurious_alphas_train.zip
rm spurious_alphas_train.zip
cd ..
