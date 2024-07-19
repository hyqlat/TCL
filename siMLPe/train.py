import argparse
import os, sys
import json
import math
import numpy as np
import copy

from config import config
from model import CFPN as Model
from datasets.h36m import H36MDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from datasets.h36m_eval import H36MEval

from test import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from progress.bar import Bar
import time


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))



def update_buffer_params(epoch, model, dataloader, config, curr_total_stage, is_cuda=True):        
    
    count = 0
    al_total_mean = 0
        
    model.eval()
    ii = 0
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(dataloader))
    for (h36m_motion_input, h36m_motion_target) in dataloader:
        #if ii >2 :
         #   break
        bt = time.time()
        if config.deriv_input:
            b,n,c = h36m_motion_input.shape
            h36m_motion_input_ = h36m_motion_input.clone()
            h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
        else:
            h36m_motion_input_ = h36m_motion_input.clone()
        
        with torch.no_grad():
            motion_pred, alpha = model(h36m_motion_input_.cuda())
            al_total_mean += torch.sum(alpha)
        count += b
        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(ii+1, len(dataloader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
        ii += 1

    bar.finish()
        
    if curr_total_stage == 0:
        al_total_mean = 1.0
    else:
        al_total_mean = 1.0 - (al_total_mean / count)
            
    al_total_mean = torch.tensor(al_total_mean, device=config.device)
    #save alpha
    print("curr stage:{}, alpha is {}".format(curr_total_stage, al_total_mean))
    model.register_buffer('al_total_mean_stage_'+str(curr_total_stage), al_total_mean)
    
    return al_total_mean

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer, epo) :
    if nb_iter > 30000:
        current_lr = 1e-4
    else:
        current_lr = 3e-4
    

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def mpjpe_error_p3d_alpha(motion_pred, motion_gt, alpha, alpha_b, curr_stage, config, model):
    """

    :param outputs:B T C22 3
    :param all_seq:
    :param dct_n:
    :param dim_used:
    :return:
    """
    n, t, n, d = motion_gt.shape

    loss_list = []
    cou = 0
    idx_beg = 0
    in_n = config.motion.h36m_input_length

    for i in range(curr_stage):
        alpha_temp = getattr(model, 'al_total_mean_stage_{}'.format(i))
        if i > 0:
            idx_beg = config.cfpn.t_short_stage[i-1]
        loss_idx = [j for j in range(idx_beg, config.cfpn.t_short_stage[i])]

        loss_temp = alpha_temp * torch.norm(motion_pred[:, loss_idx] - motion_gt[:, loss_idx], 2, dim=-1)#B T C
        loss_list.append(loss_temp)
        cou += 1
    
    #curr_stage
    if cou > 0:
        idx_beg = config.cfpn.t_short_stage[cou-1]
    loss_idx = [j for j in range(idx_beg, config.cfpn.t_short_stage[cou])]
    
    loss_temp = alpha*torch.norm(motion_pred[:, loss_idx] - motion_gt[:, loss_idx], 2, dim=-1) + alpha_b#B T C
    loss_list.append(loss_temp)#list of B T C
    loss_list = torch.cat(loss_list, dim=1)#B T C

    mean_3d_err = torch.mean(loss_list)

    return mean_3d_err

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr, curr_stage, epo) :
    alpha_sum_temp = 0
    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred, alpha_out = model(h36m_motion_input_.cuda())
    
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    ##cal alp
    if curr_stage == 0:#
        alpha = 1
        alpha_b = 0
    else:
        alpha_out_cal = alpha_out.unsqueeze(-1)
        alpha = 1 - alpha_out_cal
        alpha_b = config.cfpn.old_weight * ((1-alpha_out_cal) * torch.log(1 - alpha_out_cal) + torch.log(1 + alpha_out_cal))
        alpha_sum_temp += torch.sum(alpha)
        

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,22,3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,22,3)

    loss = mpjpe_error_p3d_alpha(motion_pred, h36m_motion_target, alpha, alpha_b, curr_stage, config, model)

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred[:,:config.cfpn.t_short_stage[curr_stage]])
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt[:,:config.cfpn.t_short_stage[curr_stage]])
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=100)#
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer, epo)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr, alpha_sum_temp

model = Model(config)
model.train()
model.cuda()

##
curr_stage = 0
##

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=1, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
print('begin training!')
nb_iter = 0
avg_loss = 0.
avg_lr = 0.
alpha_sum = 0.
bs_cou = 0.

epo = 0

while (nb_iter + 1) < config.cos_lr_total_iters:
    

    for (h36m_motion_input, h36m_motion_target) in dataloader:
        bs_cou += h36m_motion_input.shape[0]

        loss, optimizer, current_lr, alpha_sum_temp = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min, curr_stage, epo)
        avg_loss += loss
        avg_lr += current_lr

        alpha_sum += alpha_sum_temp

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss} \t epo: {epo} \t plpha_curr: {alpha_sum/bs_cou}")
            avg_loss = 0
            avg_lr = 0

            alpha_sum = 0
            bs_cou = 0 

        if (nb_iter + 1) == config.cfpn.stage_tr_epo[curr_stage]:
            model.eval()
            update_buffer_params(nb_iter, model, dataloader, config, curr_stage)
            curr_stage += 1
            model.train()


        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1
    
    epo += 1

writer.close()



