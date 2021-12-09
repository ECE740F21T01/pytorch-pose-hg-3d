#!/bin/bash

conda create -n pose-hg-3d python=3.6.13
conda activate pose-hg-3d

# Install dependencies from VIBE repo (VIBE pre-processes MPI-INF-3DHP, 3DPW datasets):
pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r lib/vibe_requirements.txt  --ignore-installed certifi

# Install dependencies of pytorch-pose-hg-3d
conda install --channel https://conda.anaconda.org/menpo opencv
conda install --channel https://conda.anaconda.org/auto progress

# Install cocoapi
sudo apt-get install -y python-setuptools
pip install Cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# Install Additonal dependencies
pip install tensorflow # tensorboard
pip install tensorboardX
pip install matplotlib
pip install scipy

# Disable cudnn for batch_norm (see [issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)):
PYTORCH=~/anaconda3/envs/pose-hg-3d/lib/python3.6/site-packages/
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
