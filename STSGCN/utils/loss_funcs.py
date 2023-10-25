#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils



def mpjpe_error(batch_pred,batch_gt): 
    
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))


def mpjpe_error_tcl(batch_pred, batch_gt, alpha, alpha_list, stage_length_list, curr_stage): 
    loss_list = []
    in_n = 0 #50?
    idx_beg = 0
    cou = 0
    wei = 24.0
    
    for i in range(curr_stage):
      alpha_temp = alpha_list[i]#第一个是0
      if i > 0:
        idx_beg = in_n + stage_length_list[i-1]
      loss_idx = [j for j in range(idx_beg, in_n+stage_length_list[i])]
      # loss_temp = (1 - alpha_temp) * torch.norm(decoder_pred[:, loss_idx] - decoder_gt[:, loss_idx], dim=3)#B T C
      loss_temp = (1 - alpha_temp) * torch.norm(batch_gt[:, loss_idx] -batch_pred[:, loss_idx],2,3)
      loss_list.append(loss_temp)
      cou += 1

    #curr_stage
    if cou > 0:
      idx_beg = in_n + stage_length_list[cou-1]

      alpha = alpha.unsqueeze(-1)
      alpha_b = wei * ((1-alpha)*torch.log(1-alpha) + torch.log(1+alpha))
    else:
      alpha = 0
      alpha_b = 0

    #print("curr_stage:", cou, " alpha:", alpha.mean())
        #print(curr_total_stage," ", cou, " ", idx_beg, " ", self.opt.t_short_stage[cou], " ", '\n\n')
    loss_idx = [j for j in range(idx_beg, in_n + stage_length_list[cou])]
    #alpha_b =  1 / (alpha.unsqueeze(-1) + 1e-15)
    # alpha = alpha.unsqueeze(-1)
    # alpha_b = wei * ((1-alpha)*torch.log(1-alpha) + torch.log(1+alpha))
    # loss_temp = (1 - alpha) * torch.norm(decoder_pred[:, loss_idx] - decoder_gt[:, loss_idx], dim=3) + alpha_b#B T C
    loss_temp = (1 - alpha) * torch.norm(batch_gt[:, loss_idx] -batch_pred[:, loss_idx],2,3) + alpha_b#B T C
    loss_list.append(loss_temp)#list of B T C
    loss_list = torch.cat(loss_list, dim=1)#B T C
    #print(alpha, alpha_b,torch.sum(loss_list), " ", loss_list.shape, torch.mean(loss_list))
    #print(torch.mean(torch.norm(p3d_out_all[:, loss_idx] - p3d_sup[:, loss_idx], dim=3)))
    loss = torch.mean(loss_list)
    return loss



def mpjpe_error_test_show(batch_pred,batch_gt): 
    #B T(25) N 3
    return torch.sum(torch.mean(torch.norm(batch_gt-batch_pred,2,3), 2), 0)#T

    
def euler_error(ang_pred, ang_gt):

    # only for 32 joints
    
    dim_full_len=ang_gt.shape[2]

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors




