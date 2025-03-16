import torch
from tqdm import tqdm

class ElasticWeightConsolidation:

    def __init__(self, model, opt, lr, weight=1000000, optimizer=None):
        self.model = model
        self.weight = weight
        self.opt = opt
        self.optimizer = optimizer

    def _update_altotalmean(self, al_total_mean, curr_stage):
        print("curr stage:{}, alpha is {}".format(curr_stage, al_total_mean))
        self.model.register_buffer('al_total_mean_stage_'+str(curr_stage), al_total_mean)
        

    def _update_fisher_params(self, dl, dim_used, t_short, curr_stage, cal_al_network):        
        grad_loss = None
        count = 0
        al_total_mean = 0
        
        self.model.eval()
        cal_al_network.eval()
        
        for i, (p3d_h36) in tqdm(enumerate(dl), total=len(dl), ncols=70):
            p3d_h36 = p3d_h36.float().to(self.opt.cuda_idx)
            bs,_,_ = p3d_h36.shape
            input = p3d_h36[:, :, dim_used].clone()
            
            with torch.no_grad():
                p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1, al_t = self.model(input)
            
                al_b, al_c, al_time = al_t.shape
                al_t = al_t.reshape([al_b, -1])#B CT
                al = cal_al_network(al_t)
            
                al_total_mean += torch.sum(al)
            
            count += input.shape[0]
        
        if curr_stage == 0:
            al_total_mean = 1
        else:
            al_total_mean = 1 - (al_total_mean / count)  
            
        my_device = torch.device(self.opt.cuda_idx)
        al_total_mean = torch.tensor(al_total_mean, device=my_device)
        self._update_altotalmean(al_total_mean, curr_stage)
        
    def register_ewc_params(self, dl, dim_used, t_short, curr_stage, cal_al_network):
        print("registering buffer!Please waiting...")
        self._update_fisher_params(dl, dim_used, t_short, curr_stage, cal_al_network)
        
    
    def smooth(self, src, sample_len, kernel_size):
        """
        data:[bs, 60, 96]
        """
        src_data = src[:, -sample_len:, :].clone()
        smooth_data = src_data.clone()
        for i in range(kernel_size, sample_len):
            smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
        return smooth_data

    def cal_loss_clip(self, p3d_out_all, p3d_sup, curr_total_stage, alpha, alpha_b):
        loss_list = []
        cou = 0
        idx_beg = 0 
        in_n = self.opt.input_n
        #pre stage
        for i in range(curr_total_stage):
            alpha_temp = getattr(self.model, 'al_total_mean_stage_{}'.format(i))
            if i > 0:
                idx_beg = in_n + self.opt.t_short_stage[i-1]
            
            loss_idx = [j for j in range(idx_beg, in_n+self.opt.t_short_stage[i])]
            loss_temp = alpha_temp * torch.norm(p3d_out_all[:, loss_idx] - p3d_sup[:, loss_idx], dim=3)#B T C
            loss_list.append(loss_temp)
            cou += 1
        
        #curr_stage
        if cou > 0:
            idx_beg = in_n+self.opt.t_short_stage[cou-1]
        loss_idx = [j for j in range(idx_beg, in_n+self.opt.t_short_stage[cou])]
        loss_temp = alpha*torch.norm(p3d_out_all[:, loss_idx] - p3d_sup[:, loss_idx], dim=3) + alpha_b#B T C
        loss_list.append(loss_temp)#list of B T C
        loss_list = torch.cat(loss_list, dim=1)#B T C
        loss = torch.mean(loss_list)
        return loss
    

    def cal_mseloss(self, p3d_h36, dim_used, p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1, t_short, mean_fs='train', alpha=1, alpha_b=0, curr_total_stage=0):
        out_n = self.opt.output_n
        in_n = seq_in = self.opt.input_n
        smooth1 = self.smooth(p3d_h36[:, :, dim_used],
                         sample_len=self.opt.kernel_size + self.opt.output_n,
                         kernel_size=self.opt.kernel_size).clone()

        smooth2 = self.smooth(smooth1,
                         sample_len=self.opt.kernel_size + self.opt.output_n,
                         kernel_size=self.opt.kernel_size).clone()

        smooth3 = self.smooth(smooth2,
                         sample_len=self.opt.kernel_size + self.opt.output_n,
                         kernel_size=self.opt.kernel_size).clone()

        p3d_sup_4 = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_3 = smooth1.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_2 = smooth2.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_1 = smooth3.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])

        b, t,_ = p3d_out_all_1.shape
        p3d_out_all_4 = p3d_out_all_4.reshape([b, t, len(dim_used) // 3, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([b, t, len(dim_used) // 3, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([b, t, len(dim_used) // 3, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([b, t, len(dim_used) // 3, 3])

        loss_idx = [i for i in range(in_n+t_short)]
        if mean_fs == 'train':
            loss_p3d_4 = self.cal_loss_clip(p3d_out_all_4, p3d_sup_4, curr_total_stage, alpha, alpha_b)
            loss_p3d_3 = self.cal_loss_clip(p3d_out_all_3, p3d_sup_3, curr_total_stage, alpha, alpha_b)
            loss_p3d_2 = self.cal_loss_clip(p3d_out_all_2, p3d_sup_2, curr_total_stage, alpha, alpha_b)
            loss_p3d_1 = self.cal_loss_clip(p3d_out_all_1, p3d_sup_1, curr_total_stage, alpha, alpha_b)
            
            loss_all_p3d = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1) / 4
        else:
            loss_p3d_4 = torch.mean(torch.mean(torch.norm(p3d_out_all_4[:, loss_idx] - p3d_sup_4[:, loss_idx], dim=3), dim=2), dim=1)
            loss_p3d_3 = torch.mean(torch.mean(torch.norm(p3d_out_all_3[:, loss_idx] - p3d_sup_3[:, loss_idx], dim=3), dim=2), dim=1)
            loss_p3d_2 = torch.mean(torch.mean(torch.norm(p3d_out_all_2[:, loss_idx] - p3d_sup_2[:, loss_idx], dim=3), dim=2), dim=1)
            loss_p3d_1 = torch.mean(torch.mean(torch.norm(p3d_out_all_1[:, loss_idx] - p3d_sup_1[:, loss_idx], dim=3), dim=2), dim=1)
            loss_all_p3d = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4#B 1
        ''''''
        return loss_all_p3d, loss_p3d_4
    
    def forward_backward_update(self, p3d_h36, dim_used, t_short, curr_total_stage, cal_al_network):
    
        self.model.train()
        cal_al_network.train()
        
        input = p3d_h36[:, :, dim_used].clone()
        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1, alpha_out_t = self.model(input)#alpha:B 1
        
        al_b, al_c, al_t = alpha_out_t.shape
        alpha_out_t = alpha_out_t.reshape([al_b, -1])#B CT
        alpha_out = cal_al_network(alpha_out_t)
        
        if curr_total_stage == 0:
            alpha = 1
            alpha_b = 0

            loss_conso = 0
        else:
            alpha_out_cal = alpha_out.unsqueeze(-1)
            alpha = 1 - alpha_out_cal
            alpha_b = self.opt.old_weight * ((1-alpha_out_cal)*torch.log(1-alpha_out_cal) + torch.log(1+alpha_out_cal))

            loss_conso = 0


        loss_p3d, loss_p3d_4 = self.cal_mseloss(p3d_h36, dim_used, p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1, t_short, 'train', alpha, alpha_b, curr_total_stage)
        
        loss = loss_p3d

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return p3d_out_all_4, loss_p3d_4, loss_conso, alpha_out#B 1

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
