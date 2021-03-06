import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import warnings
# warnings.filterwarnings("ignore")


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

'''
def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
'''

def _tranpose_and_gather_scalar(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = feat.view(feat.size(0), -1, 1)
    feat = _gather_feat(feat, ind)
    return feat

def reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr * mask.float()
    gt_regr = gt_regr * mask.float()
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    # regr_loss = nn.functional.mse_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class RegLoss(nn.Module):
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_scalar(output, ind)
    loss = reg_loss(pred, target, mask)
    return loss

class FusionLoss(nn.Module):
  def __init__(self, device, reg_weight, var_weight):
    super(FusionLoss, self).__init__()
    self.reg_weight = reg_weight
    self.var_weight = var_weight
    self.device = device
  
  def forward(self, output, mask, ind, target, gt_2d):
    pred = _tranpose_and_gather_scalar(output, ind)
    # loss = torch.FloatTensor(1)[0] * 0 # numerically unstable
    loss = torch.zeros(1)[0]
    if self.reg_weight > 0:
      loss += (self.reg_weight * reg_loss(pred, target, mask)).to('cpu')
    if self.var_weight > 0:
      #loss += (VarLoss(
        #self.device, self.var_weight)(pred, target, mask, gt_2d)[0]).to('cpu') # target for visibility
      loss += VarLoss.apply(pred, target, mask, gt_2d, self.var_weight, self.device)[0].to('cpu')
    return loss.to(self.device, non_blocking=True)


# TODO refactor this, make it neater, maybe.
class VarLoss(Function):
  """
  def __init__(self, device, var_weight):
    super(VarLoss, self).__init__()
    self.device = device
    self.var_weight = var_weight
    self.skeleton_idx = [[[0,1],    [1,2],
                          [3,4],    [4,5]],
                         [[10,11],  [11,12],
                          [13,14],  [14,15]], 
                         [[2, 6], [3, 6]], 
                         [[12,8], [13,8]]]
    self.skeleton_weight = [[1.0085885098415446, 1, 
                             1, 1.0085885098415446], 
                            [1.1375361376887123, 1, 
                             1, 1.1375361376887123], 
                            [1, 1], 
                            [1, 1]]
  """

  @staticmethod
  def forward(ctx, input, visible, mask, gt_2d, var_weight, device):
    skeleton_idx = [[[0,1],    [1,2],
                          [3,4],    [4,5]],
                         [[10,11],  [11,12],
                          [13,14],  [14,15]],
                         [[2, 6], [3, 6]],
                         [[12,8], [13,8]]]
    skeleton_weight = [[1.0085885098415446, 1,
                             1, 1.0085885098415446],
                            [1.1375361376887123, 1,
                             1, 1.1375361376887123],
                            [1, 1],
                            [1, 1]]
    xy = gt_2d.view(gt_2d.size(0), -1, 2)
    batch_size = input.size(0)
    # output = torch.FloatTensor(1) * 0 # numerically unstable
    output = torch.zeros(1)
    for t in range(batch_size):
      if mask[t].sum() == 0: # mask is the mask for supervised depth
        # xy[t] = 2.0 * xy[t] / ref.outputRes - 1
        for g in range(len(skeleton_idx)):
          E, num = 0, 0
          N = len(skeleton_idx[g])
          l = np.zeros(N)
          for j in range(N):
            id1, id2 = skeleton_idx[g][j]
            if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                      (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] * skeleton_weight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          loss = 0
          for j in range(N):
            if l[j] > 0:
              loss += (l[j] - E) ** 2 / 2. / num
          output += loss 
    output = var_weight * output / batch_size
    ctx.save_for_backward(input, visible, mask, gt_2d)
    ctx.in1 = var_weight
    ctx.in2 = device
    output = output.cuda(device, non_blocking=True)  # TODO CUDA
    return output

  @staticmethod
  def backward(ctx, grad_output):
    skeleton_idx = [[[0,1],    [1,2],
                          [3,4],    [4,5]],
                         [[10,11],  [11,12],
                          [13,14],  [14,15]],
                         [[2, 6], [3, 6]],
                         [[12,8], [13,8]]]
    skeleton_weight = [[1.0085885098415446, 1,
                             1, 1.0085885098415446],
                            [1.1375361376887123, 1,
                             1, 1.1375361376887123],
                            [1, 1],
                            [1, 1]]
    input, visible, mask, gt_2d = ctx.saved_tensors
    var_weight = ctx.in1
    device = ctx.in2
    xy = gt_2d.view(gt_2d.size(0), -1, 2)
    grad_input = torch.zeros(input.size())
    batch_size = input.size(0)
    for t in range(batch_size):
      if mask[t].sum() == 0: # mask is the mask for supervised depth
        for g in range(len(skeleton_idx)):
          E, num = 0, 0
          N = len(skeleton_idx[g])
          l = np.zeros(N)
          for j in range(N):
            id1, id2 = skeleton_idx[g][j]
            if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                      (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] * skeleton_weight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          for j in range(N):
            if l[j] > 0:
              id1, id2 = skeleton_idx[g][j]
              grad_input[t][id1] += (var_weight * \
                skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                / l[j] * (input[t, id1] - input[t, id2]) / batch_size).cpu()
              grad_input[t][id2] += (var_weight * \
                skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                / l[j] * (input[t, id2] - input[t, id1]) / batch_size).cpu()
    grad_input = grad_input.cuda(device, non_blocking=True)  # TODO CUDA
    return grad_input, None, None, None, None, None