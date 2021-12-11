# 2D Pre-Training for 3D Pose Estimation

We adopt the code from [xingyizhou/pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d), and this repository is an extention based on the PyTorch implementation for the network presented in:
> Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
> **Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach**
> ICCV 2017 ([arXiv:1704.02447](https://arxiv.org/abs/1704.02447))

## Installation
The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.4.0. The `setup.sh` includes steps to create conda environment needed, the steps are also explained here:

1. Clone the repo:
    ```
    POSE_ROOT=/path/to/clone/pytorch-pose-hg-3d
    git clone https://github.com/ECE740F21T01/pytorch-pose-hg-3d POSE_ROOT
    ```


2. Create and activate conda environment:
    ```
    conda create -n pose-hg-3d python=3.6.13
    conda activate pose-hg-3d
    ```

3. Install dependencies:
    ```
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
    ```

4. Disable cudnn for batch_norm (see [issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)):
    ```
    PYTORCH=~/anaconda3/envs/pose-hg-3d/lib/python3.6/site-packages/
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    ```

## Prepare Datasets
- Compatible datasets:
    - 2D HPE datasets:
        - MPII
        - LSP-Extended
        - FLIC-Full
    - 3D HPE datasets:
        - Human3.6M (H36M)
        - Occlusion-Person (OP)
        - MPI-INF-3DHP (MPII3D)
        - 3DPW (Implemented, but not used)
- Prepare the training data:
    - `DATA_ROOT=path/to/your/datasets/downloads`
    - MPII:
        - Download images from [MPII dataset](http://human-pose.mpi-inf.mpg.de/#download) and their [annotation](https://onedrive.live.com/?authkey=%21AKqtqKs162Z5W7g&id=56B9F9C97F261712%2110696&cid=56B9F9C97F261712) in json format (`train.json` and `val.json`) (from [Xiao et al. ECCV2018](https://github.com/Microsoft/human-pose-estimation.pytorch)).
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/mpii_human_pose_v1/annot ./data/mpii/
            ln -s ${DATA_ROOT}/mpii_human_pose_v1/images ./data/mpii/
            ```
    - LSP-Extended (We combine LSP-Extended and LSP):
        - Download LSP-Extended dataset (lspet_dataset.zip) from [official site](http://sam.johnson.io/research/lspet.html)
        - Download LSP dataset (lsp_dataset.zip) from [official site](http://sam.johnson.io/research/lspet.html)
        - linking extracted files:
            ```
            ln -s  ${DATA_ROOT}/lsp_extended/ ./data/lsp_extended
            ln -s  ${DATA_ROOT}/lsp/ ./data/lsp
            ```
    - FLIC-Full:
        - Download FLIC-Full dataset (FLIC-full.zip) from [official site](https://bensapp.github.io/flic-dataset.html)
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/FLIC-full/ ./data/flic_full
            ```
    - Human3.6M:
        - Download [Human3.6M ECCV challenge dataset](http://vision.imar.ro/human3.6m/challenge_open.php).
        - Download [meta data](https://www.dropbox.com/sh/uouev0a1ao84ofd/AADzZChEX3BdM5INGlbe74Pma/hm36_eccv_challenge?dl=0&subfolder_nav_tracking=1) (2D bounding box) of the Human3.6 dataset (from [Sun et al. ECCV 2018](https://github.com/JimmySuen/integral-human-pose)). 
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/h36m/ECCV18_Challenge ./data/h36m/
            ln -s ${DATA_ROOT}/h36m/msra_cache ./data/h36m/
            ```
    - Occlusion-Person:
        - Download Occlusion-Person dataset following instructions from their [official GitHub repo](https://github.com/zhezh/occlusion_person)
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/occlusion_person ./data/
            ```
    - MPI-INF-3DHP:
        - The following download instructions are copied from [VIBE GitHub repo](https://github.com/ECE740F21T01/VIBE/blob/master/doc/train.md)
            - MPI-3D-HP (http://gvv.mpi-inf.mpg.de/3dhp-dataset)
            - Donwload the dataset using the bash script provided by the authors. We will be using standard cameras only, so wall and ceiling cameras aren't needed. Then, run this [script](https://gist.github.com/mkocabas/cc6fe78aac51f97859e45f46476882b6) to extract frames of videos.
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/mpi_inf_3dhp ./data/
            ```
    - 3DPW:
        - Download 3DPW dataset from [official site](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html)
        - linking extracted files:
            ```
            ln -s ${DATA_ROOT}/3DPW/imageFiles ./data/3dpw/
            ln -s ${DATA_ROOT}/3DPW/sequenceFiles ./data/3dpw/
            ```
    - **Important:** After download and linking, use VIBE to preprocess MPI-INF-3DHP and 3DPW:
        ```
        source scripts/prepare_data.sh
        source scripts/prepare_training_data.sh
        ```
  
- The data folder should finally looks like this:
    ```
    ${POSE_ROOT}/data/
    ├── 3dpw
    │   ├── imageFiles
    │   └── sequenceFiles
    ├── flic_full
    │   ├── examples.mat
    │   └── images
    ├── h36m
    │   ├── ECCV18_Challenge
    │   │   ├── Scripts
    │   │   ├── Train
    │   │   └── Val
    │   └── msra_cache
    │       ├── HM36_eccv_challenge_Test_cache
    │       │   ├── HM36_eccv_challenge_Test_w256xh256_keypoint_jnt_bbox_db.pkl
    │       │   └── HM36_eccv_challenge_Test_w288xh384_keypoint_jnt_bbox_db.pkl
    │       ├── HM36_eccv_challenge_Train_cache
    │       │   ├── HM36_eccv_challenge_Train_w256xh256_keypoint_jnt_bbox_db.pkl
    │       │   └── HM36_eccv_challenge_Train_w288xh384_keypoint_jnt_bbox_db.pkl
    │       └── HM36_eccv_challenge_Val_cache
    │           ├── HM36_eccv_challenge_Val_w256xh256_keypoint_jnt_bbox_db.pkl
    │           └── HM36_eccv_challenge_Val_w288xh384_keypoint_jnt_bbox_db.pkl
    ├── lsp
    │   ├── images
    │   └── joints.mat
    ├── lsp_extended
    │   ├── images
    │   └── joints.mat
    ├── mpii
    │   ├── annot
    │   └── images
    ├── mpi_inf_3dhp
    │   ├── S1
    │   ├── S2
    |   ...
    │   ├── S8
    │   └── util
    ├── occlusion_person
    │   ├── images
    │   ├── unrealcv_train.pkl
    │   └── unrealcv_validation.pkl
    ├── vibe_data
    │   ├── J_regressor_extra.npy
    │   ├── J_regressor_h36m.npy
    │   ├── smpl_mean_params.npz
    │   ├── SMPL_NEUTRAL.pkl
    │   └── spin_model_checkpoint.pth.tar
    └── vibe_db
        ├── 3dpw_test_db.pt
        ├── 3dpw_train_db.pt
        ├── 3dpw_val_db.pt
        ├── mpii3d_train_db.pt
        └── mpii3d_val_db.pt
    ```

# Code
- Execute all experiments, e.g. `demo.py`, `main.py` from the /src/ folder.

## Demo
- Download the original pre-trained [model](https://drive.google.com/open?id=1_2CCb_qsA1egT5c2s0ABuW3rQCDOLvPq) and move it to `models`.
- Run `python demo.py --demo /path/to/image/or/image/folder [--gpus -1] [--load_model /path/to/model]`. 

`--gpus -1` is for CPU mode. 
We provide example images in `images/`. For testing your own image, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Benchmark Testing
To test our model on Human3.6 dataset run 

~~~
python main.py --exp_id test --task human3d --dataset fusion_3d --load_model ../models/fusion_3d_var.pth --test --full_test
~~~

The expected results should be 64.55mm.

## Training Your Own Models

- Stage1: Train 2D pose only. [model](https://drive.google.com/open?id=1WqW1-_gCyGTB80m9MK_KUoD0dtElEQzv), [log](https://drive.google.com/open?id=1yKwmGD4MURHnDD5536niPjxe-keY3HGs)

```
python main.py --exp_id mpii --dataset mpii
```

- Stage2: Train on 2D and 3D data without geometry loss (drop LR at 45 epochs). [model](https://drive.google.com/open?id=13d3AqzA85TSO7o1F8aq_ptnAkJ7LSp9-), [log](https://drive.google.com/open?id=18B_aOM9djCHZFlB0Rcoa6zOK1eXvsmRl)

```
python main.py --exp_id fusion_3d --task human3d --dataset fusion_3d --dataset2D mpii --dataset3D H36M --ratio_3d 1 --weight_3d 0.1 --load_model ../exp/mpii/model_last.pth --num_epoch 60 --lr_step 45
```

- Stage3: Train with geometry loss. [model](https://drive.google.com/open?id=1_2CCb_qsA1egT5c2s0ABuW3rQCDOLvPq), [log](https://drive.google.com/open?id=1hV4V74lTUd3COnoe1XMiTb8EUcyI8obN)

```
python main.py --exp_id fusion_3d_var --task human3d --dataset fusion_3d --dataset2D mpii --dataset3D H36M --ratio_3d 1 --weight_3d 0.1 --weight_var 0.01 --load_model ../models/fusion_3d.pth  --num_epoch 10 --lr 1e-4
```

### Dataset Selection and Other Input Arguments
- See detailed instructions in [new_readme.md](new_readme.md)
- See args defined in [src/lib/opts.py](src/lib/opts.py)

### 3D Exclusive Training - Stages 2 and 3, uses all 3D data
 - Stage 2:
```
python main.py --exp_id exclusive_3d_s2 --task human3d --dataset H36M --dataset3D H36M --ratio_3d 0 --weight_3d 1.0 --num_epoch 60 --lr_step 45
```

 - Stage 3:
```
python main.py --exp_id exclusive_3d_s3 --task human3d --dataset H36M --dataset3D H36M --ratio_3d 0 --weight_3d 0.1 --load_model ../exp/exclusive_3d_s2/model_best.pth  --num_epoch 10 --lr 1e-4
```

### Implementation for Huawei Ascend NPU Server
 - Control of the Apex optimizer wrapper is found in `/src/main.py` as well as `/src/lib/train.py` and `/src/lib/train_3d.py/`.

## Citation of the original paper:

    @InProceedings{Zhou_2017_ICCV,
    author = {Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
    title = {Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }

## References to other papers borrowed from for this project:

- The original code of this project is adpoted from:
   ```
    Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

    GitHub repo: https://github.com/xingyizhou/pytorch-pose-hg-3d

    > Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
    > **Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach**
    > ICCV 2017 ([arXiv:1704.02447](https://arxiv.org/abs/1704.02447))

    @InProceedings{Zhou_2017_ICCV,
    author = {Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
    title = {Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }
    ```
- The code for MPI-INF-3DHP and 3DPW dataset preparing is adpoted from:
    ```
    VIBE: Video Inference for Human Body Pose and Shape Estimation [CVPR-2020]
    
    Github repo: https://github.com/mkocabas/VIBE

    @inproceedings{kocabas2019vibe,
    title={VIBE: Video Inference for Human Body Pose and Shape Estimation},
    author={Kocabas, Muhammed and Athanasiou, Nikos and Black, Michael J.},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }
    ```
    
- The ./lib folder holds the code from VIBE repo, for the 3DPW and MPI-INF-3DHP dataset dataloader.

- For LSP-Extended Dataloader referenced: https://github.com/bmartacho/UniPose/blob/master/utils/lsp_lspet_data.py

- For calcualting depth scaling factor between gt_3d and pts (used by annotation file of src/lib/datasets/h36m.py), referenced: 
    - https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36_eccv_challenge.py#L72-L76
    - https://github.com/JimmySuen/integral-human-pose/blob/master/common/utility/utils.py#L70