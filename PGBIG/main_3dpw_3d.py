from utils import dpw3_3d as PW3_Motion3D
from model import stage_4
from utils.opt import Options
from utils import util
from utils import log
import os
import random

from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import torch.optim as optim
from tqdm import tqdm
from elastic_weight_consolidation import ElasticWeightConsolidation

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def main(opt):
    model_path = opt.ckpt + '/all_epoch'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4.CFPN(opt=opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    curr_stage = 0
    t_short = opt.t_short_stage[curr_stage]

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/all_epoch/epoch_{}.pth.tar'.format(opt.ckpt, opt.load_epoch)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len, map_location=torch.device(opt.cuda_idx))
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']

        st = 0
        for i in range(len(opt.stage_tr_epo)):
            if(start_epoch > opt.stage_tr_epo[i]):
                st += 1
        if st > 0:
            net_pred.pred_model.buffer_add(st)
        t_short = opt.t_short_stage[st]
        curr_stage = st
        net_pred.load_state_dict(ckpt['state_dict'])
        
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))


    ##model, opt, lr, weight=1000000
    ewc = ElasticWeightConsolidation(model=net_pred.pred_model, opt=opt, lr=lr_now, weight=opt.old_weight,  optimizer=optimizer)

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = PW3_Motion3D.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valid_dataset = PW3_Motion3D.Datasets(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=1,
                                  pin_memory=True)

    test_dataset = PW3_Motion3D.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=1,
                             pin_memory=True)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(ewc, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, dim_used=dim_used, t_short=t_short, curr_stage=curr_stage, cal_al_network=net_pred.cal_alpha)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used, curr_stage=curr_stage)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))

            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo, dim_used=dim_used, curr_stage=curr_stage)
            print('testing error: {:.3f}'.format(ret_test['#40ms']))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            
            if epo == opt.stage_tr_epo[curr_stage]:
                ewc.register_ewc_params(data_loader, dim_used, t_short, curr_stage, net_pred.cal_alpha)
                curr_stage += 1
                t_short = opt.t_short_stage[min(curr_stage, len(opt.t_short_stage)-1)]            
            
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

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
    ret_test = run_model(net_pred, is_train=3, data_loader=data_loader, opt=opt, dim_used=dim_used)
    ret_log = np.array(['avg'])
    head = np.array(['action'])

    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, ['test_' + k])

    log.save_csv_eval_log(opt, head, ret_log, is_create=True)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None, t_short=5, curr_stage=0, cal_al_network=None):
    if is_train == 0:
        net_pred.model.train()
    else:
        net_pred.eval()

    l_p3d = 0
    l_conso = 0
    al_total_mean = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1)*40
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.input_n

    st = time.time()
    for i, (p3d_h36) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=70):
        batch_size, seq_n, all_dim = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)
        
        if is_train == 0:
            p3d_out_all_4, loss_p3d_4, loss_conso, al = net_pred.forward_backward_update(p3d_h36, dim_used, t_short, curr_stage, cal_al_network)
            # update log values
            
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size
            if epo > opt.stage_tr_epo[0]:
                al_total_mean += torch.sum(al)
                if i % 1000 == 0:
                    print('al_curr:', torch.mean(al))
        else:#no train
            input = p3d_h36[:, :, dim_used].clone()
            p3d_out_all_4, _,_,_,_ = net_pred(input)
       

        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, all_dim//3, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim//3, 3])


        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out_4, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|loss_conso:{}|loss_p3d:{}|alpha_mean:{}'.format(i + 1, len(data_loader), time.time() - bt,                                                           time.time() - st, l_conso/n, l_p3d/n, al_total_mean/n))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret

if __name__ == '__main__':
    seed_torch(1092)
    option = Options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)
