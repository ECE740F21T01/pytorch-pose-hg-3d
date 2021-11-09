## For setting up environment and link to dataset:
Please refer to `setup.sh`

## Compatible Datasets:

2D Pretraining:
- MPII

3D Finetuning:
- Human3.6M (a.k.a. H36M)
- MPI-INF-3DHP (a.k.a. MPII3D)
- 3DPW
- Occlusion-Person (a.k.a. OcclusionPerson)

To choose which 3D dataset for evaluation and training, add this argument
```
self.parser.add_argument('--dataset3D', default = 'H36M', 
                             help = 'H36M | MPII3D | 3DPW | OcclusionPerson')
```

## References:

The original code of this project is adpoted from:

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

The code for MPI-INF-3DHP dataset preparing is adpoted from:
    
    VIBE: Video Inference for Human Body Pose and Shape Estimation [CVPR-2020]
    
    Github repo: https://github.com/mkocabas/VIBE

    @inproceedings{kocabas2019vibe,
    title={VIBE: Video Inference for Human Body Pose and Shape Estimation},
    author={Kocabas, Muhammed and Athanasiou, Nikos and Black, Michael J.},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }

    VIBE github repo references the following sources:
    "- Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/nkolot/SPIN).
    - SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
    - Some functions are borrowed from [Temporal HMR](https://github.com/akanazawa/human_dynamics).
    - Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
    - Some functions are borrowed from [Kornia](https://github.com/kornia/kornia).
    - Pose tracker is from [STAF](https://github.com/soulslicer/openpose/tree/staf)."
    
The ./lib folder holds the code from VIBE repo, for the 3DPW and MPI-INF-3DHP dataset dataloader.

