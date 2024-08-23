import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

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
    def __init__(self, root_path, data_path, size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)    
        data = np.squeeze(data) # (T S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        
        self.data_x = data
        self.data_y = data
        self.covariate = repeat_era5

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, station_id:station_id+1]
        seq_y = self.data_y[r_begin:r_end, station_id:station_id+1] # (L 1)
        t1 = self.covariate[s_begin:s_end, :, station_id:station_id+1].squeeze()
        t2 = self.covariate[r_begin:r_end, :, station_id:station_id+1].squeeze()
        seq_x = np.concatenate([t1, seq_x], axis=1) # (L 37)
        seq_y = np.concatenate([t2, seq_y], axis=1) # (L 37)
        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num



warnings.filterwarnings('ignore')
class Dataset_Feature(Dataset):
    def __init__(self, root_path, data_path,external_data_path,size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.features = features
        self.root_path = root_path
        self.external_data_path = external_data_path
        self.data_path = data_path
        if 'wind' in self.data_path:
            self.flag='wind'
        elif 'temp' in self.data_path:
            self.flag='temp'
        else:
            self.flag=None
        self.__read_data__()
        self.stations_num = self.data_x.shape[-2]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path))
        external_data = np.load(os.path.join(self.root_path, self.external_data_path))
        data = np.concatenate([external_data,data],axis=2)
        data = np.squeeze(data) # (T S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy'))
        #对大气压进行单位变换为kpa
        era5[:,-1,:,:]=era5[:,-1,:,:]/1000
        repeat_era5 = neighbor_insert(era5) # (T, 4, 9, S)
        #在此处进行数据筛选
        with open('flitered_station.txt', 'r') as f:
            station_idx=f.readlines()
        station_idx=[int(x) for x in station_idx[0].split(" ")]
        #数据筛选结束    
        repeat_era5_1 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        self.data_x = data.astype(np.float32)
        self.data_y = data.astype(np.float32)
        self.covariate = repeat_era5_1.astype(np.float32)
        del data,repeat_era5,era5,external_data,repeat_era5_1
    
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
        seq_x = np.concatenate([t1, seq_x], axis=1) # (L 37)
        seq_y = np.concatenate([t2, seq_y], axis=1) # (L 37)
        del t1,t2
        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num


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
        repeat_era5 = neighbor_insert(repeat_era5)
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

class Dataset_Fuse(Dataset):
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
        external_data = np.load(os.path.join(self.root_path, self.external_data_path))
        data = np.concatenate([external_data,data],axis=2)
        data = np.squeeze(data) # (T S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
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
        seq_x = np.concatenate([t1, seq_x], axis=1) # (L 37)
        seq_y = np.concatenate([t2, seq_y], axis=1) # (L 37)
        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num