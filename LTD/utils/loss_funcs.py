import numpy as np
import torch
from torch.autograd import Variable
from utils import data_utils


def sen_loss(outputs, all_seq, dim_used, dct_n):
    """

    :param outputs: N * (seq_len*dim_used_len)
    :param all_seq: N * seq_len * dim_full_len
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)
    dim_used = np.array(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    pred_expmap = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    targ_expmap = all_seq.clone()[:, :, dim_used]

    loss = torch.mean(torch.sum(torch.abs(pred_expmap - targ_expmap), dim=2).view(-1))
    return loss


def euler_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp

    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = pred_expmap.view(-1, 3)
    targ_expmap = targ_expmap.view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors


def mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :param data_mean:
    :param data_std:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len).clone()
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch(targ_expmap).view(-1, 3)

    pred_p3d = data_utils.expmap2xyz_torch(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_cmu(outputs, all_seq, input_n, dim_used, dct_n):
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)
    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch_cmu(targ_expmap).view(-1, 3)
    pred_p3d = data_utils.expmap2xyz_torch_cmu(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used):
    """

    :param outputs:n*66*dct_n
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_p3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err


def mpjpe_error_p3d_alpha(outputs, all_seq, dct_n, dim_used, alpha, alpha_b, curr_stage, opt, model):
    """

    :param outputs:n*66*dct_n
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_p3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    loss_list = []
    cou = 0
    idx_beg = 0
    in_n = opt.input_n
    
    #ba, ti, cd = outputs_p3d.shape#B T C
    pred_3d = outputs_p3d#.contiguous().view(-1, dim_used_len).view(-1, 3)   B T D
    targ_3d = all_seq[:, :, dim_used]#.contiguous().view(-1, dim_used_len).view(-1, 3) B T D

    for i in range(curr_stage):
        alpha_temp = getattr(model, 'al_total_mean_stage_{}'.format(i))
        if i > 0:
            idx_beg = in_n + opt.t_short_stage[i-1]#错了，实际上你还有10帧条件啊
        #print( curr_total_stage," ", i, " ", idx_beg, " ", self.opt.t_short_stage[i], " ")
        #print("curr_stage:", i, " alpha:", alpha_temp)
        #print(getattr(self.model, 'al_total_mean_stage_{}'.format(1)))
        loss_idx = [j for j in range(idx_beg, in_n+opt.t_short_stage[i])]

        loss_temp = alpha_temp * torch.norm(pred_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3) - targ_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3), 2, dim=-1)#B T C
        loss_list.append(loss_temp)
        cou += 1
    
    #curr_stage
    if cou > 0:
        idx_beg = in_n+opt.t_short_stage[cou-1]
    loss_idx = [j for j in range(idx_beg, in_n+opt.t_short_stage[cou])]
    #alpha_b =  1 / (alpha.unsqueeze(-1) + 1e-15)
    loss_temp = alpha*torch.norm(pred_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3) - targ_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3), 2, dim=-1) + alpha_b#B T C
    loss_list.append(loss_temp)#list of B T C
    loss_list = torch.cat(loss_list, dim=1)#B T C
    #print(alpha, alpha_b,torch.sum(loss_list), " ", loss_list.shape, torch.mean(loss_list))
    #print(torch.mean(torch.norm(p3d_out_all[:, loss_idx] - p3d_sup[:, loss_idx], dim=3)))
    mean_3d_err = torch.mean(loss_list)

    #pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    #targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    #mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err


def mpjpe_error_3dpw_alpha(outputs, all_seq, dct_n, dim_used, alpha, alpha_b, curr_stage, opt, model):
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used = np.array(dim_used)
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)

    loss_list = []
    cou = 0
    idx_beg = 0
    in_n = opt.input_n

    pred_3d = outputs_exp#.contiguous().view(-1, dim_used_len).view(-1, 3)   B T D
    targ_3d = all_seq[:, :, dim_used]#.contiguous().view(-1, dim_used_len).view(-1, 3) B T D

    for i in range(curr_stage):
        alpha_temp = getattr(model, 'al_total_mean_stage_{}'.format(i))
        if i > 0:
            idx_beg = in_n + opt.t_short_stage[i-1]#错了，实际上你还有10帧条件啊
        #print( curr_total_stage," ", i, " ", idx_beg, " ", self.opt.t_short_stage[i], " ")
        #print("curr_stage:", i, " alpha:", alpha_temp)
        #print(getattr(self.model, 'al_total_mean_stage_{}'.format(1)))
        loss_idx = [j for j in range(idx_beg, in_n+opt.t_short_stage[i])]

        loss_temp = alpha_temp * torch.norm(pred_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3) - targ_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3), 2, dim=-1)#B T C
        loss_list.append(loss_temp)
        cou += 1
    
    #curr_stage
    if cou > 0:
        idx_beg = in_n+opt.t_short_stage[cou-1]
    loss_idx = [j for j in range(idx_beg, in_n+opt.t_short_stage[cou])]
    #alpha_b =  1 / (alpha.unsqueeze(-1) + 1e-15)
    loss_temp = alpha*torch.norm(pred_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3) - targ_3d[:, loss_idx].contiguous().view(n, -1, dim_used_len//3, 3), 2, dim=-1) + alpha_b#B T C
    loss_list.append(loss_temp)#list of B T C
    loss_list = torch.cat(loss_list, dim=1)#B T C
    #print(alpha, alpha_b,torch.sum(loss_list), " ", loss_list.shape, torch.mean(loss_list))
    #print(torch.mean(torch.norm(p3d_out_all[:, loss_idx] - p3d_sup[:, loss_idx], dim=3)))
    mean_3d_err = torch.mean(loss_list)

    ###
    '''
    pred_3d = all_seq.clone()
    pred_3d[:, :, dim_used] = outputs_exp
    pred_3d = pred_3d.contiguous().view(-1, dim_full_len).view(-1, 3)
    targ_3d = all_seq.contiguous().view(-1, dim_full_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))
    '''
    return mean_3d_err

def mpjpe_error_3dpw(outputs, all_seq, dct_n, dim_used):
    n, seq_len, dim_full_len = all_seq.data.shape

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = all_seq.clone()
    pred_3d[:, :, dim_used] = outputs_exp
    pred_3d = pred_3d.contiguous().view(-1, dim_full_len).view(-1, 3)
    targ_3d = all_seq.contiguous().view(-1, dim_full_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err
