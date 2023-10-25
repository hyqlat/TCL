import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd
from model import *
from utils.ang2joint import *
from utils.amass_3d import *
from utils.parser import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)
model = Model(args.input_dim,args.input_n,
                           args.output_n,args.st_gcnn_dropout,args.joints_to_consider,args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to(device)        

def test():
        print('Test mode')
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'model'))) 
        model.eval()
        accum_loss=None  
        n=0
        Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
        loader_test = DataLoader(
            Dataset,
            batch_size=args.batch_size,
            shuffle =False,
            num_workers=0)
        joint_used=np.arange(4,22)
        full_joint_used=np.arange(0,22) # needed for visualization
        with torch.no_grad():
            for cnt,batch in enumerate(loader_test): 
                batch = batch.float().to(device)
                batch_dim=batch.shape[0]
                n+=batch_dim
                
                sequences_train=batch[:,0:args.input_n,joint_used,:].permute(0,3,1,2)

                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,full_joint_used,:]
                
                sequences_predict, _=model(sequences_train)
                
                sequences_predict = sequences_predict.permute(0,1,3,2)  #B T C 3
                all_joints_seq=sequences_predict_gt.clone()

                all_joints_seq[:,:,joint_used,:]=sequences_predict

                loss=mpjpe_error_test(all_joints_seq,sequences_predict_gt)*1000 # loss in milimeters
                if accum_loss is None:
                    accum_loss = loss
                else:
                    accum_loss += loss*batch_dim
        idx = [1,3,7,9,13,24]
        print('average is:\n', accum_loss[idx]/n)

if __name__ == '__main__':
    test()
    
