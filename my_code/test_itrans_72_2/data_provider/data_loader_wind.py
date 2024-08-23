import os
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import pearsonr
import warnings
import gc
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
    def __init__(self, root_path, data_path, external_data_path, size, features='MS'):
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.root_path = root_path
        self.external_data_path = external_data_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.data_x.shape[-2]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path))
        external_data = np.load(os.path.join(self.root_path, self.external_data_path))
        data = np.concatenate([external_data, data], axis=2)
        data = np.squeeze(data)  # (T, S)
        
        repeat_era5 = np.load(os.path.join(self.root_path, 'global_data.npy'))
        shape_0, shape_1, shape_2, shape_3 = repeat_era5.shape

        # 衍生特征计算
        wind_speed = np.sqrt(repeat_era5[:, 0, :, :]**2 + repeat_era5[:, 1, :, :]**2)
        wind_direction = np.degrees(np.arctan2(repeat_era5[:, 0, :, :], repeat_era5[:, 1, :, :]))
        wind_speed = wind_speed.reshape(shape_0, 1, shape_2, shape_3)
        wind_direction = wind_direction.reshape(shape_0, 1, shape_2, shape_3)

        repeat_era5 = np.concatenate([repeat_era5, wind_speed, wind_direction], axis=1)
        repeat_era5 = mean_insert(repeat_era5)  # 假设 mean_insert 是一个高效的函数
        del wind_speed,wind_direction
        
        #在此处进行数据筛选
        temp_idx=-1 if 'temp' in self.data_path else -2
        era5_temp=repeat_era5[:,2,4,:].transpose(1,0)
        temp=data[:,:,temp_idx].transpose(1,0)
        res=(era5_temp-temp).mean(1).tolist()
        res_idx=[i for i in range(len(res))]
        station_idx=[res_idx[x] for x in range(len(res)) if res[x]<4.5]
        repeat_era5=repeat_era5[:,:,:,station_idx]
        data=data[:,station_idx,:]
        del era5_temp,temp,res
        #数据筛选结束
        
        repeat_era5_1 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)

        self.data_x = data.astype(np.float32)
        self.data_y = data.astype(np.float32)
        self.covariate = repeat_era5_1.astype(np.float32)
        del data,repeat_era5,external_data,repeat_era5_1
        gc.collect()

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, station_id:station_id+1].squeeze(1)
        seq_y = self.data_y[r_begin:r_end, station_id:station_id+1].squeeze(1)  # (L, 1)

        t1 = self.covariate[s_begin:s_end, :, station_id:station_id+1].squeeze()
        t2 = self.covariate[r_begin:r_end, :, station_id:station_id+1].squeeze()

        seq_x_t = np.concatenate([t1, seq_x], axis=1)  # (L, 37)
        seq_y_t = np.concatenate([t2, seq_y], axis=1)  # (L, 37)
        del t1,t2,seq_x, seq_y
        # gc.collect()
        return seq_x_t, seq_y_t

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

