from torch.nn import Module
from torch import nn
import torch

from model import BaseModel as BaseBlock
import utils.util as util
from utils.opt import Options

class CFPN(Module):
    def __init__(self, opt):
        super(CFPN, self).__init__()
        
        self.input_n = opt.input_n
        self.output_n = opt.output_n
        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features//3
        
        self.pred_model = MultiStageModel(opt)
        self.cal_alpha = nn.Sequential(
            nn.Linear((self.input_n + self.output_n) * self.node_n, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, src):      
        output_4, output_3, output_2, output_1, output_cal_alpha = self.pred_model(src)    
        al_b, al_c, al_t = output_cal_alpha.shape
        output_cal_alpha = output_cal_alpha.reshape([al_b, -1])#B CT
        alpha = self.cal_alpha(output_cal_alpha)#B 1
        
        return output_4, output_3, output_2, output_1, alpha

class MultiStageModel(Module):
    def __init__(self, opt):
        super(MultiStageModel, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size

        self.d_model = opt.d_model
        # self.seq_in = seq_in
        self.dct_n = opt.dct_n
        # ks = int((kernel_size + 1) / 2)
        assert opt.kernel_size == 10

        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features//3

        self.encoder_layer_num = 1
        self.decoder_layer_num = 2

        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.gcn_encoder1 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.encoder_layer_num)

        self.gcn_decoder1 = BaseBlock.GCN_decoder(in_channal=self.d_model , out_channal=3,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n*2,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.decoder_layer_num)

        self.gcn_encoder2 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder2 = BaseBlock.GCN_decoder(in_channal=self.d_model , out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder3 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder3 = BaseBlock.GCN_decoder(in_channal=self.d_model , out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder4 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder4 = BaseBlock.GCN_decoder(in_channal=self.d_model , out_channal=4,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)
        
        for param_name, _ in self.named_parameters():
            print(param_name)


    def buffer_add(self, st):
        for i in range(st):
            self.register_buffer('al_total_mean_stage_'+str(i), torch.tensor(1.0))
        

    def forward(self, src):
        output_n = self.output_n
        input_n = self.input_n

        bs = src.shape[0]
        # [2000,512,22,20]
        dct_n = self.dct_n
        idx = list(range(self.kernel_size)) + [self.kernel_size -1] * output_n
        # [b,20,66]
        input_gcn = src[:, idx].clone()
        

        dct_m, idct_m = util.get_dct_matrix(input_n + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)


        # [b,20,66] -> [b,66,20]
        input_gcn_dct = torch.matmul(dct_m[:dct_n], input_gcn).permute(0, 2, 1)

        # [b,66,20]->[b,22,3,20]->[b,3,22,20]->[b,512,22,20]
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, -1, self.dct_n).permute(0, 2, 1, 3)


        #stage1
        latent_gcn_dct = self.gcn_encoder1(input_gcn_dct)

        #[b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)

        output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        #stage2
        latent_gcn_dct = self.gcn_encoder2(output_dct_1)

        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)

        output_dct_2 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        #stage3
        latent_gcn_dct = self.gcn_encoder3(output_dct_2)

        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)

        output_dct_3 = self.gcn_decoder3(latent_gcn_dct)[:, :, :, :dct_n]

        #stage4
        latent_gcn_dct = self.gcn_encoder4(output_dct_3)

        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)

        output_dct_4_t = self.gcn_decoder4(latent_gcn_dct)[:, :, :, :dct_n]
        output_dct_4 = output_dct_4_t[:, :3, :, :dct_n]

        ###
        output_cal_alpha = output_dct_4_t[:, 3, :, :dct_n]#B C T

        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)

        # [b,20 66]->[b,20 66]
        output_1 = torch.matmul(idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1))
        output_2 = torch.matmul(idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1))
        output_3 = torch.matmul(idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1))
        output_4 = torch.matmul(idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1))

        return output_4, output_3, output_2, output_1, output_cal_alpha

if __name__ == '__main__':
    option = Options().parse()
    option.d_model = 64
    model = MultiStageModel(opt=option).cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    src = torch.FloatTensor(torch.randn((32, 35, 66))).cuda()
    output, att_map,zero = model(src)
