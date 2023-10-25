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
from utils.pose3dpw3d import Pose3dPW3D
import utils.model as nnmodel
import utils.data_utils as data_utils



def main(opt):
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_out_{:d}_dctn_{:d}'.format(opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    model = nnmodel.CFPN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=69)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if opt.is_load:
        model_path_len = 'checkpoint/running_res_3dpw/saveckpt.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)

    # data loading
    print(">>> loading data")
    train_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=0,
                               dct_n=dct_n)
    dim_used = train_dataset.dim_used
    test_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=1,
                              dct_n=dct_n)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    print(">>> data loaded !")
    print(">>> test data {}".format(test_dataset.__len__()))

    ret_log = np.array([])
    head = np.array([])
    test_3d = test(test_loader, model,
                       input_n=input_n,
                       output_n=output_n,
                       is_cuda=is_cuda,
                       dim_used=dim_used,
                       dct_n=dct_n)

    ret_log = np.append(ret_log, test_3d * 1000)
    if output_n == 15:
        head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d'])
    elif output_n == 30:
        head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d', '6003d', '7003d', '8003d', '9003d',
                                    '10003d'])   
        
    # update log file
    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
    df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)        

def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()
        else:
            inputs = Variable(inputs).float()
            all_seq = Variable(all_seq).float()
        outputs,_ = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view \
            (-1, dim_full_len - 3, seq_len).transpose(1, 2)
        pred_3d = all_seq.clone()
        pred_3d[:, :, dim_used] = outputs_exp
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy() * n

        # update the training loss
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d / N


if __name__ == "__main__":
    option = Options().parse()
    main(option)
