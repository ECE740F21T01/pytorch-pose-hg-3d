'''
Modified from "src/lib/datasets/lsp_extended.py" and "src/lib/datasets/mpii.py"

By changing to load the FLIC-Full dataset .mat annotation files from data/flic_full in the __init__ function.
- The images and annotations in data/flic_full are from: https://bensapp.github.io/flic-dataset.html

Changed _get_part_info to load the correct format labels

Changed _load_image to load from path given in the .pkl file

Referenced data processing of: 
'''
import os
import torch.utils.data as data
import numpy as np
import torch
import json
import scipy.io as sio
import cv2
from PIL import Image
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds

def remove_nan_and_calculate_z(row):
    x = row[0]
    y = row[1]
    z = row[2]
    if np.isnan(x) and np.isnan(y):
        return [0, 0, 0]
    elif np.isnan(x) and (not np.isnan(y)):
        raise ValueError("both x,y should be nan at the same time", row)
        return [0, y, 0]
    elif (not np.isnan(x)) and np.isnan(y):
        raise ValueError("both x,y should be nan at the same time", row)
        return [x, 0, 0]
    else:
        return [x, y, z]

class FLICFull(data.Dataset):
    def __init__(self, opt, split):
        print('==> initializing 2D {} data.'.format(split))
        self.num_joints = 16
        self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
        self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                            [10, 15], [11, 14], [12, 13]]
        self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                    [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                    [6, 8], [8, 9]]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        self.flic_to_mpii = [12, 11, 10, 7, 8, 9, 19, 21, 18, 17, 6, 5, 4, 1, 2, 3]

        if split == "train":
            # example[7]: istrain
            idx_split = 7
        elif split == "val":
            # example[8]: istest
            idx_split = 8
        else:
            raise ValueError("Invalid Dataset Split:", split)

        self.data_path = os.path.join(opt.data_dir, 'flic_full')
        examples = sio.loadmat(os.path.join(self.data_path, 'examples.mat'))['examples'][0]
        examples_splitted = []
        for i in range(len(examples)):
            if (examples[i][idx_split][0][0] == 1):
                examples_splitted.append(examples[i])

        self.num_samples = len(examples_splitted)
        annot = {}
        tags = ['image','joints', 'center', 'scale']
        for tag in tags:
            annot[tag] = []
        tags_to_idx = {
            'image': 3,
            'joints': 2,
        }

        for idx in range(self.num_samples):
            # annot['joints']
            cur_joints = examples_splitted[idx][tags_to_idx['joints']].transpose([1, 0]) # change to shape (29, 2)
            cur_joints = np.append(cur_joints, np.ones((cur_joints.shape[0],1)), axis=1) # add z value as ones, shape become (29,3)
            cur_joints = np.apply_along_axis(remove_nan_and_calculate_z, 1, cur_joints) # remove nan, and set z to zero if joint have nan value
            cur_lms = cur_joints.transpose([0,1]) # change to shape (3, 29), save for calculating scale and center
            annot['joints'].append(cur_joints)
            
            # annot['image']
            img_path = os.path.join(self.data_path, 'images', examples_splitted[idx][tags_to_idx['image']][0])
            annot['image'].append(img_path)
            im = Image.open(img_path)
            w = im.size[0]
            h = im.size[1]
            
            # annot['center']
            center_x = (cur_lms[0][cur_lms[0] < w].max() +
                        cur_lms[0][cur_lms[0] > 0].min()) / 2
            center_y = (cur_lms[1][cur_lms[1] < h].max() +
                        cur_lms[1][cur_lms[1] > 0].min()) / 2
            annot['center'].append([center_x, center_y])

            # annot['scale']
            scale = (cur_lms[1][cur_lms[1] < h].max() -
                    cur_lms[1][cur_lms[1] > 0].min() + 4) / 368.0
            annot['scale'].append(scale)
                    
        for tag in tags:
            annot[tag] = np.array(annot[tag])

        print('Loaded 2D {} {} samples'.format(split, self.num_samples))
        self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
        self.split = split
        self.opt = opt
        self.annot = annot
  
    def _load_image(self, index):
        path = self.annot['image'][index]
        img = cv2.imread(path)
        return img
  
    def _get_part_info(self, index):
        pts = self.annot['joints'][index].copy().astype(np.float32)
        pts = pts[self.flic_to_mpii]

        c = self.annot['center'][index].copy().astype(np.float32)
        s = self.annot['scale'][index]
        c[1] = c[1] + 15 * s
        c -= 1
        s = s * 1.25
        s = s * 200
        return pts, c, s
      
    def __getitem__(self, index):
        img = self._load_image(index)
        pts, c, s = self._get_part_info(index)
        r = 0
    
        if self.split == 'train':
            sf = self.opt.scale
            rf = self.opt.rotate
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if np.random.random() <= 0.6 else 0
            s = min(s, max(img.shape[0], img.shape[1])) * 1.0
        s = np.array([s, s])
        s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)

        flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
        if flipped:
            img = img[:, ::-1, :]
            c[0] = img.shape[1] - 1 - c[0]
            pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
            for e in self.shuffle_ref:
                pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

        trans_input = get_affine_transform(
            c, s, r, [self.opt.input_h, self.opt.input_w])
        inp = cv2.warpAffine(img, trans_input, (self.opt.input_h, self.opt.input_w),
                            flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        trans_output = get_affine_transform(
            c, s, r, [self.opt.output_h, self.opt.output_w])
        out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                        dtype=np.float32)
        pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
        for i in range(self.num_joints):
            if pts[i, 0] > 0 or pts[i, 1] > 0:
                pts_crop[i] = affine_transform(pts[i], trans_output)
                out[i] = draw_gaussian(out[i], pts_crop[i], self.opt.hm_gauss) 
        
        meta = {'index' : index, 'center' : c, 'scale' : s, \
                'pts_crop': pts_crop}
        return {'input': inp, 'target': out, 'meta': meta}
    
    def __len__(self):
        return self.num_samples

    def convert_eval_format(self, pred, conf, meta):
        ret = np.zeros((pred.shape[0], pred.shape[1], 2))
        for i in range(pred.shape[0]):
            ret[i] = transform_preds(
                pred[i], meta['center'][i].numpy(), meta['scale'][i].numpy(), 
                [self.opt.output_h, self.opt.output_w])
        return ret

