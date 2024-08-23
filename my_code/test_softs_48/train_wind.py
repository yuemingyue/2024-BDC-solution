import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import warnings
import gc
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from data_provider.data_loader import Dataset_Feature,Dataset_Meteorology
from models.SOFTS_ATTN import SOFTS_ATTN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
seed_torch(8019)

def neighbor_insert(era5):
    shape_0 = era5.shape[0]
    shape_1 = era5.shape[1]
    shape_2 = era5.shape[2]
    shape_3 = era5.shape[3]
    era_insert=np.zeros((shape_0*3,shape_1,shape_2,shape_3))
    for i in range(shape_0):
        index=i*3
        now_data=era5[i]
        if i+1<shape_0:
            next_data=era5[i+1]
        else:
            next_data=era5[i]
        era_insert[index]=now_data
        era_insert[index+1]=now_data*0.7+next_data*0.3
        era_insert[index+2]=now_data*0.3+next_data*0.7
    return era_insert
    
class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path,external_data_path,size=None, features='MS',data_fliter=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_fliter=data_fliter
        self.features = features
        self.root_path = root_path
        self.external_data_path = external_data_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.data_x.shape[-2]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path))
        # print(data.shape)
        external_data = np.load(os.path.join(self.root_path, self.external_data_path))
        data = np.concatenate([external_data,data],axis=2)
        data = np.squeeze(data) # (T S)
        repeat_era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        repeat_era5 = neighbor_insert(repeat_era5)
        #在此处进行数据筛选
        if self.data_fliter:
            temp_idx=-1 if 'temp' in self.data_path else -2
            wind_idx=-2 if 'temp' in self.data_path else -1
            era5_temp=repeat_era5[:,2,4,:].transpose(1,0)
            temp=data[:,:,temp_idx].transpose(1,0)        
            res=(era5_temp-temp).mean(1).tolist()
            res_idx=[i for i in range(len(res))]
             # 使用布尔索引找出温度全零的行索引
            rows_all_zero = np.all(temp == 0, axis=1)
            temp_index = np.where(rows_all_zero)[0]
            # 使用布尔索引找出风速全零的行索引
            wind=data[:,:,wind_idx].transpose(1,0)
            rows_all_zero_wind = np.all(wind == 0, axis=1)
            wind_index = np.where(rows_all_zero_wind)[0]
            #取交集
            union_index = set(temp_index)&set(wind_index)
            station_idx=[res_idx[x] for x in range(len(res)) if -4.5<res[x]<4.5]
            repeat_era5=repeat_era5[:,:,:,station_idx]
            data=data[:,station_idx,:]
            self.station_idx=station_idx
            del era5_temp,temp,res,wind,rows_all_zero,union_index,rows_all_zero_wind,wind_index,temp_index
        #数据筛选结束
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        self.data_x = data.astype(np.float32)
        self.data_y = data.astype(np.float32)
        self.covariate = repeat_era5.astype(np.float32)
        del data,repeat_era5,external_data

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, station_id:station_id+1].squeeze(1)
        seq_y = self.data_y[r_begin:r_end, station_id:station_id+1].squeeze(1) # (L 1)
        t1 = self.covariate[s_begin:s_end, :, station_id:station_id+1].squeeze()
        t2 = self.covariate[r_begin:r_end, :, station_id:station_id+1].squeeze()

        seq_x_t = np.concatenate([t1, seq_x], axis=1) # (L 37)
        seq_y_t = np.concatenate([t2, seq_y], axis=1) # (L 37)
        del t1,t2,seq_x, seq_y
        return seq_x_t, seq_y_t

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

path="/home/mw/input/bdc_train9198/global/"
root_path=path+'global'
data_path='wind.npy'
exter_path='temp.npy'
dataset=Dataset_Meteorology(root_path,data_path,exter_path,[168,1,48],features='MS',data_fliter=False)



class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss,self).__init__()

    def forward(self,outputs,batch_y):
        return (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
        
def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def DiffLoss(y_true, y_pred):
    diff_loss=torch.tensor(0.000).to(y_true.device)
    length=12
    for i in range(6):
        y_true_diff=(y_true[:,12*i:12*(i+1)]).diff(dim=1)
        y_pred_diff=(y_pred[:,12*i:12*(i+1)]).diff(dim=1)
        diff_loss+=torch.mean((y_true_diff - y_pred_diff) ** 2)
    return diff_loss/6

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

#DATALOADER
bs=7000
shuffle_flag=True
loader=DataLoader(dataset,batch_size=bs,shuffle=shuffle_flag,num_workers=32)
#MODEL
device='cuda' if torch.cuda.is_available() else 'cpu'
enc_in,c_out,pred_len,device=37,1,24,device
model=SOFTS_ATTN(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=48,d_model=128,dropout=.05,use_norm=True,d_core=48,d_ff=160,activation='gelu',e_layers=1).to(device)
#OPTIMIZER
lr=4e-3
wd=5e-7
grad_accum_step=1
optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
#LOSS
maeloss=nn.L1Loss()
mseloss=nn.MSELoss()
loss_fn=FFTLoss()
#EPOCHS
epochs=2
#AMP
scaler = torch.cuda.amp.GradScaler()
use_amp=True
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5)
#FSNET
from tqdm import tqdm
bst_loss=9999999
bst_score=9999999
for epo in range(epochs):
    epo_loss=[]
    epo_score=[]
    for step,(batch_x,batch_y) in tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epo+1}", leave=True):
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                #FORWARD
                batch_out=model(batch_x)
                batch_out = batch_out[:, -48:,-1:]
                batch_y = batch_y[:, -48:,-1:]
                with torch.no_grad():
                    mse=mse_loss(batch_out,batch_y)
                    score=mse/torch.var(batch_y)
                loss=0.5*mseloss(batch_out,batch_y)+0.5*maeloss(batch_out,batch_y)
        else:
            batch_out=model(batch_x)
            batch_out = batch_out[:, -24:,-1:]
            batch_y = batch_y[:, -24:,-1:]
            with torch.no_grad():
                mse=mse_loss(batch_out[:, -24:,-1:],batch_y[:, -24:,-1:])
                score=mse/torch.var(batch_y[:, -24:,-1:])
            loss=0.5*mseloss(batch_out,batch_y)+0.5*maeloss(batch_out,batch_y)
        #BACKWARD
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            if (step+1)%grad_accum_step==0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epo_loss.append(mse.item())
        epo_score.append(score.item())
        if step%500==0 and step!=0:
            epo_loss_sum=sum(epo_loss)/len(epo_loss)
            epo_score_sum=sum(epo_score)/len(epo_score)
            print(f'now epoch{epo},now step{step}:mse_loss:{epo_loss_sum},score:{epo_score_sum}')
            #if step%5000==0:
                #torch.save(model.state_dict(),'chusai_trained_model_wind/attn_soft.pt')
            if epo_score_sum<bst_score:
                bst_score=epo_score_sum
                torch.save(model.state_dict(),'/home/mw/project/best_model/best_attn_soft_48.pt')
    scheduler.step()
    epo_loss_sum=sum(epo_loss)/len(epo_loss)
    epo_score_sum=sum(epo_score)/len(epo_score)
    torch.save(model.state_dict(),f'/home/mw/project/best_model/best_attn_soft_48_epo_{epo}.pt')
    print(f'now epoch{epo},now step{step}:mse_loss:{epo_loss_sum},score:{epo_score_sum}')

seed_torch(8019)