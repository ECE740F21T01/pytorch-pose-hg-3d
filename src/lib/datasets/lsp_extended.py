'''
Modified from "src/lib/datasets/mpii.py"

By changing to load the LSP-Extended and LSP dataset .mat annotation files from data/lsp_extended, data/lsp in the __init__ function.
- The images and annotations in data/lsp_extended are from: http://sam.johnson.io/research/lspet.html
- The images and annotations in data/lsp are from: http://sam.johnson.io/research/lsp.html
- load entire lsp-extended as training set
- load first half (im0001.jpg~im1000.jpg) of lsp as training set
- load second half (im1001.jpg~im2000.jpg) of lsp as validation set

Changed _get_part_info to load the correct format labels

Changed _load_image to load from path given in the .pkl file

Referenced data processing of https://github.com/bmartacho/UniPose/blob/master/utils/lsp_lspet_data.py
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

def get_lsp_joint_names():
    return [
        'rankle', # 0
        'rknee', # 1
        'rhip', # 2
        'lhip', # 3
        'lknee', # 4
        'lankle', # 5
        'rwrist', # 6
        'relbow', # 7
        'rshoulder', # 8
        'lshoulder', # 9
        'lelbow', # 10
        'lwrist', # 11
        'neck', # 12
        'headtop', # 13
    ]

def get_mpii_joint_names():
    return [
        'rankle', # 0
        'rknee', # 1
        'rhip', # 2
        'lhip', # 3
        'lknee', # 4
        'lankle', # 5
        'hip', # 6
        'thorax', # 7
        'neck', # 8
        'headtop', # 9
        'rwrist', # 10
        'relbow', # 11
        'rshoulder', # 12
        'lshoulder', # 13
        'lelbow', # 14
        'lwrist', # 15
    ]

def convert_kps_lsp_mpii(in_joints):
    src_names = eval(f'get_lsp_joint_names')()
    dst_names = eval(f'get_mpii_joint_names')()
    
    out_joints = np.zeros((len(dst_names), in_joints.shape[1]), dtype=np.float32)

    for idx, jn in enumerate(dst_names):
        # need to infer "thorax", "headtop" from other existing joints
        if jn in src_names:
            out_joints[idx, :] = in_joints[src_names.index(jn), :]
        elif jn == "thorax":
            out_joints[idx, :] = (in_joints[src_names.index("lshoulder"), :] + in_joints[src_names.index("rshoulder"), :]) * 0.5
            out_joints[idx, 2] = in_joints[src_names.index("lshoulder"), 2] * in_joints[src_names.index("rshoulder"), 2]
        elif jn == "hip":
            out_joints[idx, :] = (in_joints[src_names.index("lhip"), :] + in_joints[src_names.index("rhip"), :]) * 0.5
            out_joints[idx, 2] = in_joints[src_names.index("lhip"), 2] * in_joints[src_names.index("rhip"), 2]

    return out_joints

class LSPExtended(data.Dataset):
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
        
        if split == "train":
            # load entire lsp-extended as training set, lspnet (14,3,10000)
            self.data_path_lspet = os.path.join(opt.data_dir, 'lsp_extended')
            joints_partial_1 = sio.loadmat(os.path.join(self.data_path_lspet, 'joints.mat'))['joints']
            lms_partial_1 = joints_partial_1.transpose([2, 1, 0])
            joints_partial_1 = joints_partial_1.transpose([2, 0, 1])

            # load first half (im0001.jpg~im1000.jpg) of lsp as training set, lsp (3,14,2000)
            self.data_path_lsp = os.path.join(opt.data_dir, 'lsp')
            joints_partial_2 = sio.loadmat(os.path.join(self.data_path_lsp, 'joints.mat'))['joints']
            joints_partial_2[2] = np.logical_not(joints_partial_2[2])
            lms_partial_2 = joints_partial_2.transpose([2, 0, 1])[:1000]
            joints_partial_2 = joints_partial_2.transpose([2, 1, 0])[:1000]

            lms = np.vstack((lms_partial_1, lms_partial_2))
            joints = np.vstack((joints_partial_1, joints_partial_2))

        elif split == "val":
            # load second half (im1001.jpg~im2000.jpg) of lsp as validation set, lsp (3,14,2000)
            self.data_path_lsp = os.path.join(opt.data_dir, 'lsp')
            joints = sio.loadmat(os.path.join(self.data_path_lsp, 'joints.mat'))['joints']
            joints[2] = np.logical_not(joints[2])
            lms = joints.transpose([2, 0, 1])[1000:]
            joints = joints.transpose([2, 1, 0])[1000:]

        else:
            raise ValueError("Invalid Dataset Split:", split)

        self.num_samples = len(joints)
        annot = {}
        tags = ['image','joints', 'center', 'scale']
        for tag in tags:
            annot[tag] = []

        for idx in range(self.num_samples):
            # annot['joints']
            annot['joints'].append(joints[idx])

            # annot['image']
            if split == "train":
                if idx < 10000:
                    img_path = os.path.join(self.data_path_lspet, 'images', 'im%05d.jpg'%(idx+1))
                else:
                    img_path = os.path.join(self.data_path_lsp, 'images', 'im%04d.jpg'%(idx+1-10000))
            elif split == "val":
                img_path = os.path.join(self.data_path_lsp, 'images', 'im%04d.jpg'%(idx+1001))
            else:
                raise ValueError("Invalid Dataset Split:", split)
            annot['image'].append(img_path)
            im = Image.open(img_path)
            w = im.size[0]
            h = im.size[1]
            
            # annot['center']
            center_x = (lms[idx][0][lms[idx][0] < w].max() +
                        lms[idx][0][lms[idx][0] > 0].min()) / 2
            center_y = (lms[idx][1][lms[idx][1] < h].max() +
                        lms[idx][1][lms[idx][1] > 0].min()) / 2
            annot['center'].append([center_x, center_y])

            # annot['scale']
            scale = (lms[idx][1][lms[idx][1] < h].max() -
                    lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
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
        pts = convert_kps_lsp_mpii(pts)

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

