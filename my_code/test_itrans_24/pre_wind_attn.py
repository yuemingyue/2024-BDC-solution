
# 导入相关包
import numpy as np
import torch
import random
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from data_provider.data_loader_wind import Dataset_Meteorology
import os

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted,DataEmbedding

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR


from tqdm import tqdm
from collections import OrderedDict
from torch.optim.swa_utils import AveragedModel

#固定种子
os.environ ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'
os.environ ['inplace_normalize'] = '1'
os.environ ['fuse_relu_before_depthwise_conv'] = '1'
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
seed_torch(3047)
path="/home/mw/input/bdc_train9198/global/"
root_path=path+'global'
data_path='wind.npy'
exter_path='temp.npy'
print("读取风速数据")
dataset=Dataset_Meteorology(root_path,data_path,exter_path,[168,1,24],features='MS')
print("风速数据读取完成")

'''注意力机制'''
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
'''风速模型'''
class CNNLSTMDecoder_1(nn.Module):
    def __init__(self, d_model, pred_len, dropout, kernel_size, num_lstm_layers, d_ff):
        super(CNNLSTMDecoder_1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.lstm = nn.LSTM(input_size=d_ff, hidden_size=d_model, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_model,128,bias=True)
        self.projection = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.Mish(),
            nn.Linear(256, 92, bias=True),
            nn.Mish(),
            nn.Linear(92,pred_len,bias=True)
        )

    def forward(self, x):
        # x: [batch_size, input_size,seq_len]
        # CNN
        #对输入进行转置，将特征维度和时间维度对换，利用cnn在时间维度上进行卷积，在转成[b,i_p,s_q],，用lstm提取时序特征
        cnn_output = self.cnn(x.transpose(1, 2))  # Change from [batch_size, input_size,seq_len] to [batch_size, seq_len,input_size] for Conv1d
        cnn_output = cnn_output.transpose(1, 2)  # Change back to [batch_size, input_size,seq_len]
        # print(cnn_output.shape)
        
        # LSTM
        lstm_output, _ = self.lstm(cnn_output) #[batch_size,input_size, seq_len ]
        # print(lstm_output.shape)  #(bc,24,64)
        
        # Projection
        out_put = self.fc(lstm_output)
        projection_output = self.projection(out_put)
        
        return projection_output
        
class itransformer(nn.Module):
    def __init__(self,seq_len,pred_len,output_attention,dropout,d_model,n_heads,d_ff,activation,e_layers):
        super(itransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention =output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len,d_model,dropout)
        #attention
        self.era_attn = CBAMAttention(9,1,3)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention),d_model,n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = CNNLSTMDecoder_1(d_model, pred_len, dropout, kernel_size=3,num_lstm_layers=1, d_ff=d_ff)
    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _,_,N = x_enc.shape #[b_s,i_t,s_l]

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # 注意力机制
        enc_out = enc_out.permute(0, 2, 1)
        era_in, temp_wind = enc_out[:, :, :6 * 9], enc_out[:, :, -2:]
        # print(era_in.shape)
        era_in = era_in.reshape(-1, 64, 6, 9).permute(0, 3, 2, 1)
        era_in = self.era_attn(era_in).permute(0, 3, 1, 2).reshape(-1, 64, 6 * 9)
        enc_out = torch.cat([era_in, temp_wind], axis=-1).permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        
        dec_out = self.decoder(enc_out).permute(0, 2, 1)[:, :, :N]
 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, C]

'''指数移动平均'''
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

'''warm_up学习率调整'''
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1.0, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


'''超参数设置'''
bs=7000
shuffle_flag=True
train_loader=DataLoader(dataset,batch_size=bs,shuffle=shuffle_flag,num_workers=8)
# val_loader=DataLoader(val_dataset,batch_size=2500,shuffle=shuffle_flag,num_workers=10)
#MODEL
device='cuda' if torch.cuda.is_available() else 'cpu'
model=itransformer(seq_len=168,pred_len=24,output_attention=0,dropout=0.1,d_model=64,n_heads=1,d_ff=64,activation='gelu',e_layers=1).to(device)
#OPTIMIZER
lr=0.003
#LOSS
loss_fn=nn.MSELoss()
#EPOCHS
epochs=2
#AMP
scaler = torch.cuda.amp.GradScaler()
use_amp=True
loss_mode='feq'

optimizer=torch.optim.Adam(model.parameters(), lr=lr)
print(len(train_loader))
num_warmup_steps = len(train_loader)*epochs*0.1
num_training_steps = len(train_loader)*epochs

# 创建学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
ema = EMA(model, 0.995)
ema.register()

best_score=9999999
for epo in range(epochs):
    train_loss=[]
    train_mse=[]
    train_score=[]
    val_mse=[]
    val_score=[]
    for step,(batch_x,batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epo+1}", leave=True):
        batch_x=batch_x.float().to(device)
        batch_y=batch_y.float().to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                #FORWARD
                batch_out=model(batch_x)
                batch_out = batch_out[:, -24:,-1:]
                batch_y = batch_y[:, -24:,-1:]    
        else:
            batch_out=model(batch_x)
            batch_out = batch_out[:, -24:,-1:]
            batch_y = batch_y[:, -24:,-1:]

        if loss_mode=='mse':
            loss = loss_fn(batch_out,batch_y)
        elif loss_mode=='feq':
            loss = (torch.fft.rfft(batch_out, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
        
        #BACKWARD
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            ema.update()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            loss.backward()
            ema.update()
            optimizer.step()
            scheduler.step()
            
        pred  = batch_out.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()
        
        mse = np.mean((pred-true)**2)
        score = mse/np.var(true)
        
        train_mse.append(mse)
        train_loss.append(loss.item())
        train_score.append(score)
        if step%500==0 and step!=0:
            loss_sum=sum(train_loss)/len(train_loss)
            mse_sum = sum(train_mse)/len(train_mse)
            score_sum = sum(train_score)/len(train_score)
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print(f'now epoch{epo},now step{step}:loss:{loss_sum},mse:{mse_sum},score:{score_sum}')
            if score_sum<best_score:
                print("save_model:",score_sum)
                best_score=score_sum
                torch.save(model.state_dict(),'/home/mw/project/best_model/best_test_24_itrans_model_wind_attn.pt')
    loss_sum=sum(train_loss)/len(train_loss)
    mse_sum = sum(train_mse)/len(train_mse)
    score_sum = sum(train_score)/len(train_score)
    print(f'now epoch{epo}:loss:{loss_sum},mse:{mse_sum},score:{score_sum}')
    # torch.save(model.state_dict(),f'home/mw/project/best_model/test_24_itrans_model_wind_attn_epo_{epo}.pt')
    print(f'now epoch{epo}:loss:{loss_sum},mse:{mse_sum},score:{score_sum}')