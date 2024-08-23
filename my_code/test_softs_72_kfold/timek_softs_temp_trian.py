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
from models.Dain_SOFTS_TEMP import Dain_SOFTS_TEMP
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

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
        else:
            self.station_idx=[i for i in range(self.stations_num)]
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

class Dataset_Meteorology_split_by_time(Dataset):
    def __init__(self, root_path, data_path,external_data_path,time_idx,size=None, features='MS',data_fliter=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_fliter=data_fliter
        self.features = features
        self.root_path = root_path
        self.external_data_path = external_data_path
        self.data_path = data_path
        self.time_idx=time_idx
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
            del era5_temp,temp,res,wind,rows_all_zero,union_index,rows_all_zero_wind,wind_index,temp_index
        #数据筛选结束
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        data=data[self.time_idx,:,:]
        repeat_era5=repeat_era5[self.time_idx,:,:]
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
data_path='temp.npy'
exter_path='wind.npy'
dataset=Dataset_Meteorology(root_path,data_path,exter_path,[168,1,72],features='MS',data_fliter=True)


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
from torch.optim.lr_scheduler import LambdaLR
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1.0, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

kfold=5
#### DATALOADER
bs=7000
shuffle_flag=True
loader=DataLoader(dataset,batch_size=bs,shuffle=shuffle_flag,num_workers=30)
#MODEL
device='cuda' if torch.cuda.is_available() else 'cpu'
model=Dain_SOFTS_TEMP(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=72,d_model=168+72,dropout=.2,use_norm=True,d_core=24,d_ff=256,activation='gelu',e_layers=1).to(device)
#OPTIMIZER
lr=3e-3
wd=5e-5
grad_accum_step=1
optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
#LOSS
maeloss=nn.SmoothL1Loss()
mseloss=nn.MSELoss()
loss_fn=FFTLoss()
#EPOCHS
epochs=3
#AMP
scaler = torch.cuda.amp.GradScaler()
use_amp=True
num_warmup_steps = len(loader)*epochs*0.2
num_training_steps = len(loader)*epochs
# 创建学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
time_index=[i for i in range(17544)]
skf = KFold(n_splits = 5,random_state = None,shuffle = False)
for num_fold,(train_time_idx,valid_time_idx) in enumerate(skf.split(time_index,time_index)):
    train_time_idx=train_time_idx.tolist()
    valid_time_idx=valid_time_idx.tolist()
    print(f"***********************************fold:{num_fold+1}数据加载**************************************")
    train_set=Dataset_Meteorology_split_by_time(root_path,data_path,exter_path,train_time_idx,[168,1,72],features='MS',data_fliter=True)
    #valid_set=Dataset_Meteorology_split_by_station(root_path,data_path,exter_path,valid_station_idx,[168,1,72],features='MS',data_fliter=False)
    train_loader = DataLoader(train_set,batch_size=bs,shuffle=shuffle_flag,num_workers=30)
    print(f"**********************************fold:{num_fold+1}模型加载*****************************************")
    # 初始化模型 配置
    model = Dain_SOFTS_TEMP(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=72,d_model=128,dropout=.1,use_norm=True,d_core=72,d_ff=256,activation='gelu',e_layers=1).to(device)
    swa_model =AveragedModel(model).to(device)
    #AMP
    scaler = torch.cuda.amp.GradScaler()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=3e-6)
    num_warmup_steps = len(loader)*epochs*0.2
    num_training_steps = len(loader)*epochs
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
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
            regularization_strength = 1e-5  # 这个系数可以根据需要调整
            loss=0.5*loss_fn(batch_out,batch_y)+0.5*maeloss(batch_out,batch_y)
            #BACKWARD
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                ema.update()
                optimizer.step()
                scheduler.step()
                
            pred = batch_out.detach().cpu().numpy()
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
                    print(f"lr:{param_group['lr']}")
                print(f'now fold {num_fold+1} epoch{epo},now step{step},loss:{loss_sum},mse:{mse_sum},score:{score_sum}' )
                if score_sum<best_score:
                    print("save_model:",score_sum)
                    best_score=score_sum
                    #torch.save(model.state_dict(),f'softs_temp_kfold_checkpoint/best_softs_temp_model_fold_{num_fold+1}_epo_{epo}.pt')
            scheduler.step()           
        if epo>=1:
            print("swa_model")
            swa_model.update_parameters(model)
        # epoch 级别指标计
        loss_sum=sum(train_loss)/len(train_loss)
        mse_sum = sum(train_mse)/len(train_mse)
        score_sum = sum(train_score)/len(train_score)
        
        # 保存epoch 级别 指标
        train_loss_fold.append(loss_sum)
        train_mse_fold.append(mse_sum)
        train_score_fold.append(score_sum)
        torch.save(model.state_dict(),f'/home/mw/project/best_model/softs_temp_model_fold_{num_fold+1}_epo_{epo}.pt')
    # fold 级别指标
    print(f'now fold {num_fold+1} mse:{np.mean(train_mse_fold)},score:{np.mean(train_score_fold)}' )
    #保存swa模型
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    #torch.save(swa_model.state_dict(), f'softs_temp_kfold_checkpoint/swa_softs_temp_model_fold_{num_fold+1}_epo_{epo}.pt')
    del train_set,train_loader