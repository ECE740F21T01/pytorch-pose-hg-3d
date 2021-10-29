#!/bin/bash

conda create -n pose-hg-3d python=3.6
conda activate pose-hg-3d

# conda install pytorch=0.4.1 cuda90 torchvision -c pytorch
conda install --channel https://conda.anaconda.org/menpo opencv
conda install --channel https://conda.anaconda.org/auto progress

# Disable cudnn for batch_norm (see issue):
# /home/liyao/HDD/anaconda3/envs/pose-hg-3d/lib/python3.6/site-packages/torch
PYTORCH=/home/liyao/HDD/anaconda3/envs/pose-hg-3d/lib/python3.6/site-packages/
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py

pip install tensorflow # tensorboard
pip install matplotlib
pip install scipy

# cocoapi
sudo apt-get install -y python-setuptools
pip install Cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# datasets:
# MPII:
ln -s /data/keith/datasets/mpii_human_pose_v1/annot ~/HDD/projects/ECE740/pytorch-pose-hg-3d/data/mpii/
ln -s /data/keith/datasets/mpii_human_pose_v1/images ~/HDD/projects/ECE740/pytorch-pose-hg-3d/data/mpii/
