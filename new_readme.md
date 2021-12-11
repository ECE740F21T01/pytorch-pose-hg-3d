## Datatset Selection Arguments:

- Dataset argument options:
  - For Stage 1 (2D pre-training), use the following options to select the dataset to use:
    - `--dataset <2d-dataset-name>`

  - For both Stage 2 and 3 (train on both 2D and 3D datasets), use the following options to select the dataset to use:
    - `--dataset fusion_3d --dataset2D <2d-dataset-name> --dataset3D <3d-dataset-name>`

  - For 3D Exclusive Training in Stages 2 and 3, uses all 3D data, use the following options to select the dataset to use:
    - `--dataset <3d-dataset-name>`

  - Note:
    - `<2d-dataset-name>` can be selected from one of: `mpii | lsp_extended | flic_full`
    - `<3d-dataset-name>` can be selected from one of: `H36M | MPII3D | 3DPW | OcclusionPerson`
  
  - Data size cap:
    - To use/remove 3D dataset size limit when using `fusion_3d`, use the `--ratio_3d` argument.
      - By default, `--ratio_3d` have default value set to 1. 
      - Using any value <= 0 will remove the size cap.
      - `--ratio_3d 1` will limit `3D dataset size` to be `1 * 2D dataset size`. Or replace 1 with any value > 0.

## Other Changes From Original Code:

- Other input argument changes:
  - added `--random_seed`, default to 0
    - added `--grad_clip`, default to None (turned off), typical value would be 5.0 (does not work on NPU)
- Issues resolved:
  - resolved NaN loss issue, by changing `torch.FloatTensor()` to `torch.zeros()` in file "src/lib/models/losses.py".
    - torch.FloatTensor() is numerically unstable and causes NaN in loss value, according to https://github.com/kevinzakka/pytorch-goodies/issues/8
