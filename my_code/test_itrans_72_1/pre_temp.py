
# 导入相关包
import numpy as np
import torch
import random
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from data_provider.data_loader_temp import Dataset_Meteorology
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
data_path='temp.npy'
exter_path='wind.npy'
print("读取温度数据")
dataset=Dataset_Meteorology(root_path,data_path,exter_path,[168,1,72],features='MS')
print("温度数据读取完成")


'''温度模型'''
class itransformer(nn.Module):
    def __init__(self, seq_len,pred_len,output_attention,dropout,d_model,n_heads,d_ff,activation,e_layers):
        super(itransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention =output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len,d_model,dropout)
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
        # Decoder
        self.projection = nn.Linear(d_model+168,d_ff, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(d_ff, 256, bias=True),
            nn.Mish(),
            nn.Linear(256, 92, bias=True),
            nn.Mish(),
            nn.Linear(92,pred_len,bias=True)
        )
        
    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out  =torch.concat((x_enc.permute(0,2,1),enc_out),dim=-1)
        dec_out = self.projection(enc_out)
        dec_out = self.fc(dec_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
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
device='cuda' if torch.cuda.is_available() else 'cpu'
lr=0.006
epochs=3
kfold =5

# 生成所有index
all_index = list(range(len(dataset)))
skf = KFold(n_splits = kfold,random_state = None,shuffle = False)

# 随机权重平均SWA,实现更好的泛化
# swa_model = AveragedModel(model).to(device)
for num_fold,(train_idx,valid_idx) in enumerate(skf.split(all_index,all_index)):
    print(f"***********************************fold:{num_fold+1}数据加载**************************************")
    train_set = Subset(dataset,train_idx)
    valid_set = Subset(dataset,valid_idx)
    train_loader = DataLoader(train_set,batch_size=bs,shuffle=shuffle_flag,num_workers=8)
    valid_loader= DataLoader(valid_set,batch_size=bs,shuffle=shuffle_flag,num_workers=8)
    print(f"**********************************fold:{num_fold+1}模型加载*****************************************")
    # 初始化模型 配置
    model =itransformer(seq_len=168,pred_len=72,output_attention=0,dropout=0.1,d_model=128,n_heads=1,d_ff=128,activation='gelu',e_layers=1).to(device)
    swa_model = AveragedModel(model).to(device)
    #AMP
    scaler = torch.cuda.amp.GradScaler()
    use_amp= True
    loss_mode='feq'
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=3e-5)
    num_warmup_steps = len(train_loader)*epochs*0.2
    num_training_steps = len(train_loader)*epochs
    loss_fn=nn.MSELoss()
    loss_fn_mae = nn.L1Loss(reduction='mean')
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    ema = EMA(model, 0.999)
    ema.register()
    
    print(f"************************************fold:{num_fold+1}开始训练******************************************")
    # 指标
    train_loss_fold=[]
    train_mse_fold=[]
    train_score_fold=[]
    
    # 训练
    for epo in range(epochs):
        best_score=9999999
        train_loss=[]
        train_mse=[]
        train_score=[]
        
        for step,(batch_x,batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training fold {num_fold+1} epoch {epo+1}", leave=True):
            model.train()
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    #FORWARD
                    batch_out=model(batch_x)
                    batch_out = batch_out[:, -72:,-1:]
                    batch_y = batch_y[:, -72:,-1:]    
            else:
                batch_out=model(batch_x)
                batch_out = batch_out[:, -72:,-1:]
                batch_y = batch_y[:, -72:,-1:]
    
            # 计算 L1 正则化项
            l1_penalty = sum(torch.norm(param, 1) for param in model.parameters())
            regularization_strength = 3e-5 
            
            if loss_mode=='mse':
                loss = loss_fn(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode=='feq':
                loss = (torch.fft.rfft(batch_out, dim=1)-torch.fft.rfft(batch_y,dim=1)).abs().mean()*0.4+0.6*loss_fn(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode=='mix':
                loss = loss_fn(batch_out,batch_y)*0.5+0.5*loss_fn_mae(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode == 'base':
                loss = loss_fn(batch_out,batch_y)
                
            #BACKWARD
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                ema.update()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                loss.backward()
                ema.update()
                optimizer.step()
                scheduler.step()
                
            pred = batch_out.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            mse = np.mean((pred-true)**2)
            score = mse/np.var(true)*10
            
            train_mse.append(mse)
            train_loss.append(loss.item())
            train_score.append(score)
            if step%500==0 and step!=0:
                loss_sum=sum(train_loss)/len(train_loss)
                mse_sum = sum(train_mse)/len(train_mse)
                score_sum = sum(train_score)/len(train_score)

                for param_group in optimizer.param_groups:
                    print(f"lr:{param_group['lr']}")
                print(f'now fold {num_fold+1} epoch{epo},now step{step},loss:{loss_sum},mse:{mse_sum},score:{score_sum}' )
                if score_sum<best_score:
                    print("save_model:",score_sum)
                    best_score=score_sum
                    torch.save(model.state_dict(),f'/home/mw/project/best_model/best_test_72_itrans1_model_temp_{num_fold+1}.pt')
                    
        if epo>=1:
            # print("swa_model")
            swa_model.update_parameters(model)
        # epoch 级别指标计
        loss_sum=sum(train_loss)/len(train_loss)
        mse_sum = sum(train_mse)/len(train_mse)
        score_sum = sum(train_score)/len(train_score)
        
        # 保存epoch 级别 指标
        train_loss_fold.append(loss_sum)
        train_mse_fold.append(mse_sum)
        train_score_fold.append(score_sum)
        print(f'now epoch{epo}:loss:{loss_sum},mse:{mse_sum},score:{score_sum}')
    
    # fold 级别指标
    print(f'now fold {num_fold+1} mse:{np.mean(train_mse_fold)},score:{np.mean(train_score_fold)}' )
    # eval_fold(model,valid_loader,num_fold)
    
    # 保存fold 级别模型
    torch.save(model.state_dict(),f'/home/mw/project/best_model/test_72_itrans1_model_temp_fold_{num_fold+1}.pt')
    break

    # #保存swa模型
    # new_state_dict = OrderedDict()
    # for k, v in swa_model.state_dict().items():
    #     if k != 'n_averaged':
    #         name = k[7:]  # remove 'module.' prefix
    #         new_state_dict[name] = v
    # torch.save(new_state_dict, f'swa_temp_model_{num_fold+1}.pt')