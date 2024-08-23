import os
import numpy as np
import random
import torch
from my_code.models.Dain_SOFTS_TEMP import Dain_SOFTS_TEMP
from my_code.models.Dain_SOFTS_WIND import Dain_SOFTS_WIND

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
        era_insert[index+1]=now_data*0.7+next_data*0.3
        era_insert[index+2]=now_data*0.3+next_data*0.7
    return era_insert

def pre(func,inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'
    args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 24,
        'enc_in': 37,
        'd_model': 64,
        'n_heads': 1,
        'e_layers': 1,
        'd_ff': 64,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)
    
    test_data_root_path = inputs
    for i in range(2):
        if i == 0:
            #读取数据部分
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
            external_data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            data = np.concatenate([external_data,data],axis=3)
            #读取协变量数据
            N, L, S, _ = data.shape # 72, 168, 60
            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy")) 
            #数据进行插值
            repeat_era5 = np.stack([func(cenn_era5_data[i]) for i in range(len(cenn_era5_data))],axis=0) # (N, L, 4, 9, S)
            C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
            covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
            data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
            C = C1 + 2
            data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
            data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
            data = data.reshape(N * S, L,-1)
            data = torch.tensor(data).float().cuda() # (N * S, L, C)
            #加载模型部分
            model=Dain_SOFTS_TEMP(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=72,d_model=128,dropout=.1,use_norm=True,d_core=72,d_ff=256,activation='gelu',e_layers=1).cuda()
            checkpoint_dir='/home/mw/project/best_model/'
            model.eval()
            seeds=[8019]
            fold=[f for f in range(1,6)]
            epo=[0,1]
            seed_output=[]
            for seed in seeds:
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                fold_output=[]
                for f in fold:
                    epo_output=[]
                    for e in epo:
                        model_dir=checkpoint_dir+f'softs_temp_model_fold_{f}_epo_{e}.pt'
                        model.load_state_dict(torch.load(model_dir))
                        #预测结果
                        for n in range(2):        
                            with torch.no_grad():
                                outputs = model(data)[:, :, -1:].detach().cpu().numpy()
                        epo_output.append(outputs)
                        del outputs
                    epo_output=sum(epo_output)/len(epo_output)
                    fold_output.append(epo_output)
                fold_output=sum(fold_output)/len(fold_output)
                seed_output.append(fold_output)
            outputs=sum(seed_output)/len(seed_output)
            P = outputs.shape[1]
            forecast = outputs.reshape(N, S, P, 1) # (N, S, P, 1)
            temp_forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 1)
            #保存结果
            #np.save(os.path.join(save_path, "temp_predict.npy"), forecast)
        if i == 1:
            #读取数据部分
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy")) # (N, L, S, 1)
            external_data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))
            data = np.concatenate([external_data,data],axis=3)
            #读取协变量数据
            N, L, S, _ = data.shape # 72, 168, 60
            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy")) 
            #数据进行插值
            repeat_era5 = np.stack([func(cenn_era5_data[i]) for i in range(len(cenn_era5_data))],axis=0) # (N, L, 4, 9, S)
            C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
            covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
            data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
            C = C1 + 2
            data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
            data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
            data = data.reshape(N * S, L,-1)
            data = torch.tensor(data).float().cuda() # (N * S, L, C)
            model=Dain_SOFTS_WIND(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=72,d_model=168+72,dropout=.2,use_norm=True,d_core=72,d_ff=256,activation='gelu',e_layers=1).cuda()
            checkpoint_dir='/home/mw/project/best_model/'
            seeds = [8019]
            model.eval()
            fold=[f for f in range(1,6)]
            epo=[2]
            seed_output = []
            for seed in seeds:
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                fold_output=[]
                for f in fold:
                    epo_output=[]
                    for e in epo:
                        model_dir=checkpoint_dir+f'softs_wind_model_fold_{f}_epo_{e}.pt'
                        model.load_state_dict(torch.load(model_dir))
                        #预测结果
                        for n in range(2):
                            with torch.no_grad():
                                outputs = model(data)[:, :, -1:].detach().cpu().numpy()
                        epo_output.append(outputs)
                        del outputs
                    epo_output=sum(epo_output)/len(epo_output)
                    fold_output.append(epo_output)
                fold_output = sum(fold_output) / len(fold_output)
                seed_output.append(fold_output)
            outputs=sum(seed_output)/len(seed_output)
            P = outputs.shape[1]
            forecast = outputs.reshape(N, S, P, 1) # (N, S, P, 1)
            wind_forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 1)
            #保存结果
            #np.save(os.path.join(save_path, "wind_predict.npy"), forecast)

    return  temp_forecast,wind_forecast 




