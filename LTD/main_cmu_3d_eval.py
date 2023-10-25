#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import utils as utils
from utils.opt import Options
from utils.cmu_motion_3d import CMU_Motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils



def main(opt):
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}_dctn{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n

    model = nnmodel.CFPN(input_feature=dct_n, hidden_feature=opt.linear_size,
                        p_dropout=opt.dropout, num_stage=opt.num_stage, node_n=75)

    if is_cuda:
        model.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if opt.is_load:
        model_path_len = 'checkpoint/running_res_cmu/saveckpt.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'],strict=False)

    # data loading
    print(">>> loading data")
    train_dataset = CMU_Motion3D(path_to_data=opt.data_dir_cmu, actions=opt.actions, input_n=input_n, output_n=output_n,
                                 split=0, dct_n=opt.dct_n)
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean
    dim_used = train_dataset.dim_used

    acts = data_utils.define_actions_cmu(opt.actions)
    test_data = dict()
    for act in acts:
        test_dataset = CMU_Motion3D(path_to_data=opt.data_dir_cmu, actions=act, input_n=input_n, output_n=output_n,
                                    split=1, data_mean=data_mean, data_std=data_std, dim_used=dim_used, dct_n=dct_n)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    print(">>> data loaded !")
    print(">>> test data {}".format(test_dataset.__len__()))

    ret_log = np.array([])
    head = np.array([])
    test_3d_temp = np.array([])
    test_3d_head = np.array([])
    for act in acts:
        test_3d = test(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                               dim_used=dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, test_3d)
        head = np.append(head,
                         [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
        if output_n > 10:
            head = np.append(head, [act + '3d560', act + '3d1000'])
    ret_log = np.append(ret_log, test_3d_temp)
    head = np.append(head, test_3d_head)

    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
    df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)

def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

        outputs,_ = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used = np.array(dim_used)
        dim_used_len = len(dim_used)

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_3d = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view \
            (-1, dim_used_len, seq_len).transpose(1, 2)
        pred_3d = all_seq.clone()
        dim_used = np.array(dim_used)

        # deal with joints at same position
        joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        pred_3d[:, :, dim_used] = outputs_3d
        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy() * n

        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d / N

if __name__ == "__main__":
    option = Options().parse()
    main(option)
