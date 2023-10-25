from utils import dpw3_3d as PW3_Motion3D
from model import stage_4
from utils.opt import Options
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

def eval(opt):
    print('>>> create models')
    net_pred = stage_4.CFPN(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # load model
    model_path_len = './{}/saveckpt.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'], strict = False)

    
    dataset = PW3_Motion3D.Datasets(opt=opt, split=2)
    dim_used = dataset.dim_used
    data_loader = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=1,
                                  pin_memory=True)
    # do test


    ret_test = run_model(net_pred, data_loader=data_loader, opt=opt, dim_used=dim_used)
    ret_log = np.array(['avg'])
    head = np.array(['action'])

    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, ['test_' + k])

    log.save_csv_eval_log(opt, head, ret_log, is_create=True)

def run_model(net_pred, data_loader=None, opt=None, dim_used=None):

    net_pred.eval()
    titles = (np.array(range(opt.output_n)) + 1)*40
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.input_n
    for i, (p3d_h36) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=70):
        # print(i)
        batch_size, seq_n, all_dim = p3d_h36.shape
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)        
        input = p3d_h36[:, :, dim_used].clone()
        p3d_out_all_4, _,_,_,_ = net_pred(input)
       
        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:]
        #p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, all_dim//3, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim//3, 3])

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
