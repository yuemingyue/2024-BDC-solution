import os
import numpy as np
import random
import torch
from my_code.models import iTransformer_48
from my_code.models.SOFTS import SOFTS_MLP
from my_code.models.SOFTS_ATTN import SOFTS_ATTN

'''111,222'''
def mean_insert1(era5):
    shape_0 = era5.shape[0]
    shape_1 = era5.shape[1]
    shape_2 = era5.shape[2]
    shape_3 = era5.shape[3]
    era_insert = np.zeros((shape_0 * 3, shape_1, shape_2, shape_3))
    for i in range(shape_0):
        index = i * 3
        now_data = era5[i]
        if i + 1 < shape_0:
            next_data = era5[i + 1]
        else:
            next_data = era5[i]
        era_insert[index] = now_data
        era_insert[index + 1] = now_data
        era_insert[index + 2] = now_data
    return era_insert

'''112,223'''
def mean_insert2(era5):
    shape_0 = era5.shape[0]
    shape_1 = era5.shape[1]
    shape_2 = era5.shape[2]
    shape_3 = era5.shape[3]
    era_insert = np.zeros((shape_0 * 3, shape_1, shape_2, shape_3))
    for i in range(shape_0):
        index = i * 3
        now_data = era5[i]
        if i + 1 < shape_0:
            next_data = era5[i + 1]
        else:
            next_data = era5[i]
        era_insert[index] = now_data
        era_insert[index + 1] = now_data
        era_insert[index + 2] = next_data
    return era_insert

'''11(1+2)/2,22(2+3)/2'''
def mean_insert3(era5):
    shape_0 = era5.shape[0]
    shape_1 = era5.shape[1]
    shape_2 = era5.shape[2]
    shape_3 = era5.shape[3]
    era_insert = np.zeros((shape_0 * 3, shape_1, shape_2, shape_3))
    for i in range(shape_0):
        index = i * 3
        now_data = era5[i]
        if i + 1 < shape_0:
            next_data = era5[i + 1]
        else:
            next_data = era5[i]
        era_insert[index] = now_data
        era_insert[index+1]=now_data*0.7+next_data*0.3
        era_insert[index+2]=now_data*0.3+next_data*0.7
    return era_insert

def pre(inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'
    fix_seed = 3047
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 48,
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
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))  # (N, L, S, 1)
            external_data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)
        else:
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            external_data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)
        N, L, S, _ = data.shape  # 72, 168, 60
        '''itrans'''
        if i == 0:
            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
            # repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1)  # (N, L, 4, 9, S)

            cenn_era5_data_1 = np.stack([mean_insert1(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            cenn_era5_data_2 = np.stack([mean_insert2(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            cenn_era5_data_3 = np.stack([mean_insert3(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)

            C1 = cenn_era5_data_1.shape[2] * cenn_era5_data_1.shape[3]
            covariate_1 = cenn_era5_data_1.reshape(cenn_era5_data_1.shape[0], cenn_era5_data_1.shape[1], -1,
                                            cenn_era5_data_1.shape[4])  # (N, L, C1, S)

            covariate_2 = cenn_era5_data_2.reshape(cenn_era5_data_2.shape[0], cenn_era5_data_2.shape[1], -1,
                                            cenn_era5_data_2.shape[4])  # (N, L, C1, S)

            covariate_3 = cenn_era5_data_3.reshape(cenn_era5_data_3.shape[0], cenn_era5_data_3.shape[1], -1,
                                            cenn_era5_data_3.shape[4])  # (N, L, C1, S)

            data = data.transpose(0, 1, 3, 2)  # (N, L, 1, S)
            C = C1 + 2
            data1 = np.concatenate([covariate_1, data], axis=2)  # (N, L, C, S)
            data1 = data1.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data1 = data1.reshape(N * S, L, C)
            data_itrans1 = torch.tensor(data1).float().cuda()  # (N * S, L, C)

            data2 = np.concatenate([covariate_2, data], axis=2)  # (N, L, C, S)
            data2 = data2.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data2 = data2.reshape(N * S, L, C)
            data_itrans2 = torch.tensor(data2).float().cuda()  # (N * S, L, C)

            data3 = np.concatenate([covariate_3, data], axis=2)  # (N, L, C, S)
            data3 = data3.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data3 = data3.reshape(N * S, L, C)
            data_itrans3 = torch.tensor(data3).float().cuda()  # (N * S, L, C)
        else:
            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
            shape_0 = cenn_era5_data.shape[0]
            shape_1 = cenn_era5_data.shape[1]
            shape_2 = cenn_era5_data.shape[2]
            shape_3 = cenn_era5_data.shape[3]
            shape_4 = cenn_era5_data.shape[4]
            '''衍生特征'''
            # 风速
            wind_speed = np.sqrt(cenn_era5_data[:, :, 0, :, :] ** 2 + cenn_era5_data[:, :, 1, :, :] ** 2)
            # 风向
            wind_direction = np.degrees(np.arctan2(cenn_era5_data[:, :, 0, :, :], cenn_era5_data[:, :, 1, :, :]))
            # 气压
            # pressure_gradient = cenn_era5_data[:, :, 3, :, :] / 9.81

            wind_speed = wind_speed.reshape(shape_0, shape_1, 1, shape_3, shape_4)
            wind_direction = wind_direction.reshape(shape_0, shape_1, 1, shape_3, shape_4)
            # pressure_gradient = pressure_gradient.reshape(shape_0, shape_1,1, shape_3, shape_4)

            cenn_era5_data = np.concatenate((cenn_era5_data, wind_speed, wind_direction), axis=2)

            # repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1)  # (N, L, 4, 9, S)
            cenn_era5_data_1 = np.stack([mean_insert1(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            cenn_era5_data_2 = np.stack([mean_insert2(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            cenn_era5_data_3 = np.stack([mean_insert3(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)

            C1 = cenn_era5_data_1.shape[2] * cenn_era5_data_1.shape[3]
            covariate_1 = cenn_era5_data_1.reshape(cenn_era5_data_1.shape[0], cenn_era5_data_1.shape[1], -1,
                                                   cenn_era5_data_1.shape[4])  # (N, L, C1, S)

            covariate_2 = cenn_era5_data_2.reshape(cenn_era5_data_2.shape[0], cenn_era5_data_2.shape[1], -1,
                                                   cenn_era5_data_2.shape[4])  # (N, L, C1, S)

            covariate_3 = cenn_era5_data_3.reshape(cenn_era5_data_3.shape[0], cenn_era5_data_3.shape[1], -1,
                                                   cenn_era5_data_3.shape[4])  # (N, L, C1, S)

            data = data.transpose(0, 1, 3, 2)  # (N, L, 1, S)
            C = C1 + 2
            data1 = np.concatenate([covariate_1, data], axis=2)  # (N, L, C, S)
            data1 = data1.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data1 = data1.reshape(N * S, L, C)
            data_itrans1 = torch.tensor(data1).float().cuda()  # (N * S, L, C)

            data2 = np.concatenate([covariate_2, data], axis=2)  # (N, L, C, S)
            data2 = data2.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data2 = data2.reshape(N * S, L, C)
            data_itrans2 = torch.tensor(data2).float().cuda()  # (N * S, L, C)

            data3 = np.concatenate([covariate_3, data], axis=2)  # (N, L, C, S)
            data3 = data3.transpose(0, 3, 1, 2)  # (N, S, L, C)
            data3 = data3.reshape(N * S, L, C)
            data_itrans3 = torch.tensor(data3).float().cuda()  # (N * S, L, C)

        '''softs'''

        if i == 0:
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))  # (N, L, S, 1)
            external_data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)
        else:
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            external_data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)
        N, L, S, _ = data.shape  # 72, 168, 60
        cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
        # repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1)  # (N, L, 4, 9, S)
        cenn_era5_data_1 = np.stack([mean_insert1(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
        cenn_era5_data_2 = np.stack([mean_insert2(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
        cenn_era5_data_3 = np.stack([mean_insert3(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)

        C1 = cenn_era5_data_1.shape[2] * cenn_era5_data_1.shape[3]
        covariate_1 = cenn_era5_data_1.reshape(cenn_era5_data_1.shape[0], cenn_era5_data_1.shape[1], -1,
                                               cenn_era5_data_1.shape[4])  # (N, L, C1, S)

        covariate_2 = cenn_era5_data_2.reshape(cenn_era5_data_2.shape[0], cenn_era5_data_2.shape[1], -1,
                                               cenn_era5_data_2.shape[4])  # (N, L, C1, S)

        covariate_3 = cenn_era5_data_3.reshape(cenn_era5_data_3.shape[0], cenn_era5_data_3.shape[1], -1,
                                               cenn_era5_data_3.shape[4])  # (N, L, C1, S)

        data = data.transpose(0, 1, 3, 2)  # (N, L, 1, S)
        C = C1 + 2
        data1 = np.concatenate([covariate_1, data], axis=2)  # (N, L, C, S)
        data1 = data1.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data1 = data1.reshape(N * S, L, C)
        data_softs1 = torch.tensor(data1).float().cuda()  # (N * S, L, C)

        data2 = np.concatenate([covariate_2, data], axis=2)  # (N, L, C, S)
        data2 = data2.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data2 = data2.reshape(N * S, L, C)
        data_softs2 = torch.tensor(data2).float().cuda()  # (N * S, L, C)

        data3 = np.concatenate([covariate_3, data], axis=2)  # (N, L, C, S)
        data3 = data3.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data3 = data3.reshape(N * S, L, C)
        data_softs3 = torch.tensor(data3).float().cuda()  # (N * S, L, C)

        if i==0:
            model_itrans = iTransformer_48.Model(args).cuda()
            model_itrans.eval()
            model_soft_0=SOFTS_MLP(seq_len=168,pred_len=48,d_model=96,dropout=.05,use_norm=True,d_core=64,d_ff=128,activation='gelu',e_layers=1).cuda()
            model_soft_0.eval()
            model_soft_1 = SOFTS_MLP(seq_len=168,pred_len=48,d_model=96,dropout=.05,use_norm=True,d_core=64,d_ff=128,activation='gelu',e_layers=1).cuda()
            model_soft_1.eval()
        else:
            model_itrans = iTransformer_48.Model1(args).cuda()
            model_itrans.eval()
            model_soft=SOFTS_ATTN(seq_len=168,enc_in=38,enc_d_model=64,enc_d_core=24,pred_len=48,d_model=128,dropout=.05,use_norm=True,d_core=48,d_ff=160,activation='gelu',e_layers=1).cuda()
            model_soft.eval()
        if i == 0:
            model_soft_0.load_state_dict(torch.load("/home/mw/project/best_model/test_48_softs_model_temp_epo_0.pt"))
            model_soft_1.load_state_dict(torch.load("/home/mw/project/best_model/test_48_softs_model_temp_epo_1.pt"))
            model_itrans.load_state_dict(torch.load("/home/mw/project/best_model/best_test_48_itrans_model_temp.pt"))

        else:
            model_soft.load_state_dict(torch.load("/home/mw/project/best_model/best_attn_soft_48_epo_1.pt"))
            model_itrans.load_state_dict(torch.load("/home/mw/project/best_model/best_test_48_itrans_model_wind.pt"))
        if i == 0:
            '''softs的温度占大头'''
            for n in range(2):
                with torch.no_grad():
                    outputs_softs1 = 0.5*model_soft_0(data_softs1)+0.5*model_soft_1(data_softs1)
                    outputs_itrans1 = model_itrans(data_itrans1)
    
                    outputs_softs2 = 0.5*model_soft_0(data_softs2)+0.5*model_soft_1(data_softs2)
                    outputs_itrans2 = model_itrans(data_itrans2)
    
                    outputs_softs3 = 0.5*model_soft_0(data_softs3)+0.5*model_soft_1(data_softs3)
                    outputs_itrans3 = model_itrans(data_itrans3)
                    
                outputs_softs1 = outputs_softs1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                outputs_itrans1 = outputs_itrans1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)

                outputs_softs2 = outputs_softs2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                outputs_itrans2 = outputs_itrans2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)

                outputs_softs3 = outputs_softs3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                outputs_itrans3 = outputs_itrans3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)

            P = outputs_itrans1.shape[1]

            forecast_softs1 = outputs_softs1.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans1 = outputs_itrans1.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs1 = forecast_softs1.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans1 = forecast_itrans1.transpose(0, 2, 1, 3)  # (N, P, S, 1)

            forecast_softs2 = outputs_softs2.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans2 = outputs_itrans2.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs2 = forecast_softs2.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans2 = forecast_itrans2.transpose(0, 2, 1, 3)  # (N, P, S, 1)

            forecast_softs3 = outputs_softs3.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans3 = outputs_itrans3.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs3 = forecast_softs3.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans3 = forecast_itrans3.transpose(0, 2, 1, 3)  # (N, P, S, 1)

            temp_forecast = (forecast_softs1*0.3+forecast_itrans1 * 0.7+forecast_softs2 * 0.3 + forecast_itrans2 * 0.7+forecast_softs3 * 0.3 + forecast_itrans3 * 0.7)/3
        else:
            '''itrans的风速占大头'''
            for n in range(2):
                with torch.no_grad():
                    outputs_softs1 = model_soft(data_softs1)
                    outputs_itrans1 = model_itrans(data_itrans1)
                    outputs_softs1 = outputs_softs1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                    outputs_itrans1 = outputs_itrans1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
    
                    outputs_softs2 = model_soft(data_softs2)
                    outputs_itrans2 = model_itrans(data_itrans2)
                    outputs_softs2 = outputs_softs2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                    outputs_itrans2 = outputs_itrans2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)

                    outputs_softs3 = model_soft(data_softs3)
                    outputs_itrans3 = model_itrans(data_itrans3)
                    outputs_softs3 = outputs_softs3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
                    outputs_itrans3 = outputs_itrans3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)

            P = outputs_itrans1.shape[1]

            forecast_softs1 = outputs_softs1.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans1 = outputs_itrans1.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs1 = forecast_softs1.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans1 = forecast_itrans1.transpose(0, 2, 1, 3)  # (N, P, S, 1)

            forecast_softs2 = outputs_softs2.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans2 = outputs_itrans2.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs2 = forecast_softs2.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans2 = forecast_itrans2.transpose(0, 2, 1, 3)  # (N, P, S, 1)

            forecast_softs3 = outputs_softs3.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_itrans3 = outputs_itrans3.reshape(N, S, P, 1)  # (N, S, P, 1)
            forecast_softs3 = forecast_softs3.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            forecast_itrans3 = forecast_itrans3.transpose(0, 2, 1, 3)  # (N, P, S, 1)
            
            wind_forecast = (forecast_softs1 * 0.3+forecast_itrans1 * 0.7+ forecast_softs2 * 0.3+forecast_itrans2 * 0.7+forecast_softs3 * 0.3+forecast_itrans3 * 0.7)/3

    return temp_forecast,wind_forecast


