import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

def mean_insert(era5):
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
        era_insert[index+1]=now_data
        era_insert[index+2]=next_data
    return era_insert
    
class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path,external_data_path,size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
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


        

        repeat_era5 = mean_insert(repeat_era5)
        #在此处进行数据筛选
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
        
        station_idx=[res_idx[x] for x in range(len(res)) if -4.5<res[x]<4.5 and x not in union_index]
        repeat_era5=repeat_era5[:,:,:,station_idx]
        data=data[:,station_idx,:]
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