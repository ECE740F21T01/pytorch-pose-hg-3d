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
pip install tensorboardX
pip install matplotlib
pip install scipy

# cocoapi
sudo apt-get install -y python-setuptools
pip install Cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# datasets:
# MPII:
ln -s /data/keith/datasets/mpii_human_pose_v1/annot /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/mpii/
ln -s /data/keith/datasets/mpii_human_pose_v1/images /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/mpii/
# H36M:
ln -s /data/liyao/datasets/h36m/ECCV18_Challenge /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/h36m/
ln -s /data/liyao/datasets/h36m/msra_cache /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/h36m/
# MPI-INF-3DHP:
ln -s /data/liyao/datasets/mpi_inf_3dhp /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/
# 3DPW:
ln -s /data/keith/datasets/3DPW/imageFiles /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/3dpw/
ln -s /data/keith/datasets/3DPW/sequenceFiles /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/3dpw/
# Occlusion-Person
ln -s /data/keith/datasets/occlusion_person /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/
# LSP-Extended
ln -s  /data/keith/datasets/lsp_extended/ /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/lsp_extended
ln -s  /data/keith/datasets/lsp/ /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/lsp
# FLIC-Full
ln -s 

# Install additional envs from VIBE:
# pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r lib/vibe_requirements.txt  --ignore-installed certifi

# VIBE prepares MPI-INF-3DHP:
# or skip these two lines by copying from /data/liyao/projects/ECE740/pytorch-pose-hg-3d/data/vibe_db/* to your pytorch-pose-hg-3d/data/vibe_db
source scripts/prepare_data.sh
source scripts/prepare_training_data.sh