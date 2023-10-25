from asyncio import current_task
import os
import sys

from sklearn.covariance import log_likelihood
sys.path.append(os.path.abspath('./'))
from utils import h36motion3d as datasets
from model import stage_4
from utils.opt import Options
from utils import log
from einops import rearrange

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4.CFPN(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    #load model
    #model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    model_path_len = './{}/saveckpt.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'], strict=False)

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

    data_loader = {}
    for act in acts:
        dataset = datasets.Datasets(opt=opt, split=2, actions=act)
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=1,
                             pin_memory=True)
    #do test
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test = run_model(net_pred, data_loader=data_loader[act], opt=opt)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)

def run_model(net_pred, data_loader=None, opt=None):
    net_pred.eval()

    titles = (np.array(range(opt.output_n)) + 1)*40
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    for i, (p3d_h36) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=70):
        batch_size, seq_n, _ = p3d_h36.shape
        n += batch_size
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)

        input = p3d_h36[:, :, dim_used].clone()
        p3d_out_all_4, _,_,_,_ = net_pred(input)
        
        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:]
        p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, 32, 3])


        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        
    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    for j in range(out_n):
        ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret

if __name__ == '__main__':
    option = Options().parse()
    eval(option)

