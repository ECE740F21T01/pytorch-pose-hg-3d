## For setting up environment and link to dataset:
Please refer to `setup.sh`

## Compatible Datasets:

2D Pretraining:
- MPII
- LSP-Extended
- FLIC-Full

3D Finetuning:
- Human3.6M (a.k.a. H36M)
- MPI-INF-3DHP (a.k.a. MPII3D)
- 3DPW
- Occlusion-Person (a.k.a. OcclusionPerson)

## Updates
- Note on dataset argument options:
    - To do pretraining on 2D dataset only, or finetuning with 3D dataset only, use `--dataset xxx`:
    - When choosing opt.dataset as "fusion_3d", need to specify which 2D Dataset and which 3D dataset to use for training and evaluation (for evalaution only dataset3D is used), using `--dataset fusion_3d --dataset2D xxx --dataset3D xxx`
    - Also, notice these options that are relavent for `--dataset fusion_3d` only: `dataset2D, dataset3D, ratio_3d`

- Other changes:
    - Data size cap:
        - To use/remove 3D dataset size limit when using `fusion_3d`, use the `--ratio_3d` argument.
            - By default, `--ratio_3d` have default value set to 0, i.e. using any value <= 0 will remove the size cap
            - `--ratio_3d 1` will limit `3D dataset size` to be `1 * 2D dataset size`. Or replace 1 with any value > 0.
    - Using only 3D dataset
        - To load 2D pretrained model, and then finetune with 3D dataset only, use the `--dataset` to choose one of the 3D datasets.
            - e.g. using OcclusionPerson only, `python main.py --exp_id try_op_3d --task human3d --dataset OcclusionPerson --weight_3d 0.1 --load_model ../exp/mpii/model_last.pth --num_epoch 60 --lr_step 45`
            - Instead of finetuning, to train with 3D dataset from scratch, just remove the `--load_model` part.
    - resolved NaN loss issue, by changing `torch.FloatTensor()` to `torch.zeros()` in file "src/lib/models/losses.py".
        - torch.FloatTensor() is numerically unstable and causes NaN in loss value, according to https://github.com/kevinzakka/pytorch-goodies/issues/8
    - added `--random_seed`, default to 0
    - added `--grad_clip`, default to off

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

The code for MPI-INF-3DHP and 3DPW dataset preparing is adpoted from:
    
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

For LSP-Extended Dataloader referenced: https://github.com/bmartacho/UniPose/blob/master/utils/lsp_lspet_data.py