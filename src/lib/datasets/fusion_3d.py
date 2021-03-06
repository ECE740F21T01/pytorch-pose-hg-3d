import torch.utils.data as data
import numpy as np
import torch

from .mpii import MPII
from .lsp_extended import LSPExtended
from .flic_full import FLICFull

from .h36m import H36M
from .mpii3d import MPII3D
from .threedpw import ThreeDPW
from .occlusion_person import OcclusionPerson

class Fusion3D(data.Dataset):
  def __init__(self, opt, split):
    self.opt = opt
    self.ratio3D = opt.ratio_3d
    self.split = split
    
    if opt.dataset3D == "MPII3D":
        self.dataset3D = MPII3D(opt, split)
    elif opt.dataset3D == "3DPW":
        self.dataset3D = ThreeDPW(opt, split)
    elif opt.dataset3D == "OcclusionPerson":
        self.dataset3D = OcclusionPerson(opt, split)
    elif opt.dataset3D == "H36M":
        self.dataset3D = H36M(opt, split)
    else:
        raise ValueError("Unrecognized dataset3D:", opt.dataset3D)
    
    if self.split == 'train':
      if opt.dataset2D == "mpii":
          self.dataset2D = MPII(opt, split)
      elif opt.dataset2D == "lsp_extended":
          self.dataset2D = LSPExtended(opt, split)
      elif opt.dataset2D == "flic_full":
          self.dataset2D = FLICFull(opt, split)
      else:
          raise ValueError("Unrecognized dataset2D:", opt.dataset2D)
      self.nImages2D = len(self.dataset2D)
      if self.ratio3D <= 0:
          self.nImages3D = len(self.dataset3D)
      else:
          self.nImages3D = min(len(self.dataset3D), 
                              int(self.nImages2D * self.ratio3D))
    else:
      self.nImages3D = len(self.dataset3D)
      self.nImages2D = 0
    
    self.num_joints = self.dataset3D.num_joints
    self.num_eval_joints = self.dataset3D.num_eval_joints
    self.acc_idxs = self.dataset3D.acc_idxs
    self.edges = self.dataset3D.edges
    self.edges_3d = self.dataset3D.edges_3d
    self.shuffle_ref = self.dataset3D.shuffle_ref
    self.mean = self.dataset3D.mean
    self.std = self.dataset3D.std
    self.convert_eval_format = self.dataset3D.convert_eval_format
    print('#Images2D {}, #Images3D {}'.format(self.nImages2D, self.nImages3D))
    if self.nImages2D > 0:
        print("dataset2D: ", self.dataset2D.__class__.__name__)
    if self.nImages3D > 0:
        print("dataset3D: ", self.dataset3D.__class__.__name__)

  def __getitem__(self, index):
    if index < self.nImages3D:
      return self.dataset3D[index]
    else:
      ret = self.dataset2D[index - self.nImages3D]
      reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
      reg_ind = np.zeros((self.num_joints), dtype=np.int64)
      reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
      gt_3d = np.zeros((self.num_eval_joints, 3), dtype=np.float32)
      pts_crop = ret['meta']['pts_crop']
      for i in range(self.num_joints):
        pt = pts_crop[i]
        if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
          and pt[1] < self.opt.output_h:
          reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                      pt[0] * self.num_joints + i # note transposed
          reg_target[i] = 1 # here reg_target serves as visibility
      ret['reg_target'] = reg_target
      ret['reg_ind'] = reg_ind
      ret['reg_mask'] = reg_mask
      ret['meta']['gt_3d'] = gt_3d
      return ret

    
  def __len__(self):
    return self.nImages2D + self.nImages3D