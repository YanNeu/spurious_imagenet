#!/bin/bash

# Setup conda environment
conda env create -f reqs.yml
conda activate spurious_imagenet

cd utils
wget https://github.com/MadryLab/robustness/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
pip install -e robustness-master