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

