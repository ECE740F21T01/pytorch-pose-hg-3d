'''
Modified from "src/lib/datasets/h36m.py"
By changing to load the 3DPW (threedpw) dataset from "data/vibe_db/3dpw_train_db.pt" or "data/vibe_db/3dpw_val_db.pt"
in the __init__ function.

Changed _get_part_info to load the correct format labels

Changed _load_image to load from path given in the .pt file
'''
import os
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import pickle
import joblib
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds
from pathlib import Path
from utils.image import compute_similarity_transform


class ThreeDPW(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing 3D {} data.'.format(split))
    self.num_joints = 16
    self.num_eval_joints = 17
    self.h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
    self.mpii_to_h36m = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, \
                         13, 14, 15, 12, 11, 10]
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]
    self.shuffle_ref_3d = [[3, 6], [2, 5], [1, 4], 
                          [16, 13], [15, 12], [14, 11]]
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.edges_3d = [[3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], \
                     [0, 7], [7, 8], [8, 10],\
                     [16, 15], [15, 14], [14, 8], [8, 11], [11, 12], [12, 13]]
    self.mean_bone_length = 4000
    # normalize to mean and std of Imagenet
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
    self.split = split
    self.opt = opt

    ann_path = os.path.join(
      self.opt.data_dir, 'vibe_db', '3dpw_{}_db.pt'.format(self.split))
    self.annot = joblib.load(ann_path)
    self.annot = [dict(zip(self.annot,t)) for t in zip(*(self.annot).values())] # convert dict of lists, to list of dicts
    # keys in dataset: dict_keys(['vid_name', 'frame_id', 'joints3D', 'joints3D_absolute', 'joints2D', 'shape', 'pose', 'bbox', 'img_name', 'valid'])

    # dowmsample validation data
    self.idxs = np.arange(len(self.annot)) if split == 'train' \
                else np.arange(0, len(self.annot), 1 if opt.full_test else 10)
    self.num_samples = len(self.idxs)
    print('Loaded 3D {} {} samples'.format(split, self.num_samples))

  def _load_image(self, index):
    #path = Path(os.path.join(self.opt.data_dir, '..', self.annot[index]['img_name']))
    path = Path(os.path.join('..', self.annot[index]['img_name']))
    img = cv2.imread(str(path))
    return img
  
  def _get_part_info(self, index):
    # keys in dataset: dict_keys(['vid_name', 'frame_id', 'joints3D', 'joints3D_absolute', 'joints2D', 'shape', 'pose', 'bbox', 'img_name', 'valid'])
    # bbox = [c_x, c_y, w, h]
    # change to use 3dpw format.
    ann = self.annot[self.idxs[index]]
    gt_3d = np.array(ann['joints3D'], np.float32) # relative joint coordinates, relative to pelvis, in mpii format (16 joints)
    gt_3d = gt_3d[self.mpii_to_h36m][:17] # relative joint coordinates, relative to pelvis, converted to H36M format (17 joints)
    pts = np.array(ann['joints3D_image'], np.float32) # j3d_image is [image_coord_x, image_coord_y, depth-root depth], in mpii format (16 joints)
    c = np.array([ann['bbox'][0], ann['bbox'][1]], dtype=np.float32) # c_x, c_y for bbox
    s = max(ann['bbox'][2], ann['bbox'][3]) # w, h for bbox
    
    # to get the correct "depth-root" value for pts, using similarity transform to get depth_scale
    # and then set pts[2] = gt_3d[2] * depth_scale
    # Refence: https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36_eccv_challenge.py#L72-L76
    from_pose = gt_3d[:,0:2][self.h36m_to_mpii]
    target_pose = pts[:,0:2]
    _, _, _, depth_scale, _ = compute_similarity_transform(target_pose, from_pose, compute_optimal_scale=True)
    pts[:,2] = gt_3d[:,2][self.h36m_to_mpii] * depth_scale
    
    return gt_3d, pts, c, s
      
  def __getitem__(self, index):
    if index == 0 and self.split == 'train':
      self.idxs = np.random.choice(
        self.num_samples, self.num_samples, replace=False)
    img = self._load_image(index)
    gt_3d, pts, c, s = self._get_part_info(index)
    
    r = 0

    if self.split == 'train':
      sf = self.opt.scale
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      # rf = self.opt.rotate
      # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
      #    if np.random.random() <= 0.6 else 0

    flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
    if flipped:
      img = img[:, ::-1, :]
      c[0] = img.shape[1] - 1 - c[0]
      gt_3d[:, 0] *= -1
      pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
      for e in self.shuffle_ref_3d:
        gt_3d[e[0]], gt_3d[e[1]] = gt_3d[e[1]].copy(), gt_3d[e[0]].copy()
      for e in self.shuffle_ref:
        pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
    
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    s = np.array([s, s])
    s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)
    
    trans_input = get_affine_transform(
      c, s, r, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    trans_output = get_affine_transform(
      c, s, r, [self.opt.output_w, self.opt.output_h])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
    reg_ind = np.zeros((self.num_joints), dtype=np.int64)
    reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
      pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
      if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
        and pt[1] < self.opt.output_h:
        pts_crop[i] = pt
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)
        reg_target[i] = pts[i, 2] / s[0] # assert not self.opt.fit_short_side
        reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                     pt[0] * self.num_joints + i # note transposed
        
        reg_mask[i] = 1

    meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
            'gt_3d': gt_3d, 'pts_crop': pts_crop}

    return {'input': inp, 'target': out, 'meta': meta, 
            'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask}
    
  def __len__(self):
    return self.num_samples


  def convert_eval_format(self, pred):
    pred_h36m = pred[self.mpii_to_h36m]
    pred_h36m[7] = (pred_h36m[0] + pred_h36m[8]) / 2
    pred_h36m[9] = (pred_h36m[8] + pred_h36m[10]) / 2
    sum_bone_length = self._get_bone_length(pred_h36m)
    mean_bone_length = self.mean_bone_length
    pred_h36m = pred_h36m * mean_bone_length / sum_bone_length
    return pred_h36m

  def _get_bone_length(self, pts):
    sum_bone_length = 0
    pts = np.concatenate([pts, (pts[14] + pts[11])[np.newaxis, :] / 2])
    for e in self.edges_3d:
      sum_bone_length += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
    return sum_bone_length