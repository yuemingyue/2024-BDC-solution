import os
import numpy as np
import random
import torch
import gc
from my_code.models import iTransformer_72
from my_code.models import itransformer_temp
from my_code.models import itransformer_wind
from my_code.test_24 import pre as pre_24
from my_code.test_48 import pre as pre_48
from my_code.test_softs_72 import pre as pre_softs_72

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

def repeat_insert(era5):
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
        era_insert[index+2]=now_data
    return era_insert

def linear_insert(era5):
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
        era_insert[index+1]=now_data+(1/3)*(next_data-now_data)
        era_insert[index+2]=now_data+(2/3)*(next_data-now_data)
    return era_insert

def smooth_insert(era5):
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

def invoke(inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'
    fix_seed = 3047
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    temp_args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 72,
        'enc_in': 37,
        'd_model': 128,
        'n_heads': 1,
        'e_layers': 1,
        'd_ff': 128,
        'dropout': 0.1,
        'output_attention': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }

    wind_args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 72,
        'enc_in': 37,
        'd_model': 64,
        'n_heads': 1,
        'e_layers': 1,
        'd_ff': 128,
        'dropout': 0.1,
        'output_attention': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    temp_args = Struct(**temp_args)
    wind_args = Struct(**wind_args)


    test_data_root_path = inputs
    chusai_temp_24,chusai_wind_24 = pre_24(inputs)
    chusai_temp_48,chusai_wind_48 = pre_48(inputs)
    fgg_temp_1, fgg_wind_1 = pre_softs_72(neighbor_insert,inputs)
    fgg_temp_2, fgg_wind_2 = pre_softs_72(repeat_insert, inputs)
    fgg_temp_3, fgg_wind_3 = pre_softs_72(linear_insert, inputs)
    fgg_temp_4, fgg_wind_4 = pre_softs_72(smooth_insert, inputs)

    for i in range(2):
        '''itrans'''
        if i == 0:
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
            external_data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)  # (T S)
        else:
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            external_data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))
            data = np.concatenate([external_data, data], axis=3)
            data = np.squeeze(data)  # (T S)
        N, L, S, _ = data.shape # 72, 168, 60
        if i==0:
            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
            # repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)

            repeat_era5_1 = np.stack([neighbor_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_2 = np.stack([repeat_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_3 = np.stack([linear_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_4 = np.stack([smooth_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)

            C1 = repeat_era5_1.shape[2] * repeat_era5_1.shape[3]
            covariate_1 = repeat_era5_1.reshape(repeat_era5_1.shape[0], repeat_era5_1.shape[1], -1,
                                                repeat_era5_1.shape[4])  # (N, L, C1, S)

            covariate_2 = repeat_era5_2.reshape(repeat_era5_2.shape[0], repeat_era5_2.shape[1], -1,
                                                repeat_era5_2.shape[4])  # (N, L, C1, S)

            covariate_3 = repeat_era5_3.reshape(repeat_era5_3.shape[0], repeat_era5_3.shape[1], -1,
                                                repeat_era5_3.shape[4])  # (N, L, C1, S)
            covariate_4 = repeat_era5_4.reshape(repeat_era5_4.shape[0], repeat_era5_4.shape[1], -1,
                                                repeat_era5_4.shape[4])  # (N, L, C1, S)
            data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
            C = C1 + 2

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

            wind_speed = wind_speed.reshape(shape_0, shape_1,1, shape_3, shape_4)
            wind_direction = wind_direction.reshape(shape_0, shape_1,1, shape_3, shape_4)
            # pressure_gradient = pressure_gradient.reshape(shape_0, shape_1,1, shape_3, shape_4)

            cenn_era5_data = np.concatenate((cenn_era5_data, wind_speed, wind_direction), axis=2)

            repeat_era5_1 = np.stack([neighbor_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_2 = np.stack([repeat_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_3 = np.stack([linear_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)
            repeat_era5_4 = np.stack([smooth_insert(cenn_era5_data[i]) for i in range(len(cenn_era5_data))], axis=0)

            C1 = repeat_era5_1.shape[2] * repeat_era5_1.shape[3]
            covariate_1 = repeat_era5_1.reshape(repeat_era5_1.shape[0], repeat_era5_1.shape[1], -1,
                                                repeat_era5_1.shape[4])  # (N, L, C1, S)

            covariate_2 = repeat_era5_2.reshape(repeat_era5_2.shape[0], repeat_era5_2.shape[1], -1,
                                                repeat_era5_2.shape[4])  # (N, L, C1, S)

            covariate_3 = repeat_era5_3.reshape(repeat_era5_3.shape[0], repeat_era5_3.shape[1], -1,
                                                repeat_era5_3.shape[4])  # (N, L, C1, S)
            covariate_4 = repeat_era5_4.reshape(repeat_era5_4.shape[0], repeat_era5_4.shape[1], -1,
                                                repeat_era5_4.shape[4])  # (N, L, C1, S)
            data = data.transpose(0, 1, 3, 2)  # (N, L, 1, S)
            C = C1 + 2

        if i==0:
            model = iTransformer_72.Model(temp_args).cuda()
            model2 = iTransformer_72.Model(temp_args).cuda()
            model_ilstm = itransformer_temp.itransformer(seq_len=168,pred_len=72,output_attention=0,dropout=0.2,d_model=64,n_heads=1,d_ff=128,activation='gelu',e_layers=1).cuda()
        if i==1:
            model = iTransformer_72.Model1(wind_args).cuda()
            model2 = iTransformer_72.Model1(wind_args).cuda()
            model_ilstm = itransformer_wind.itransformer_lstm(seq_len=168, pred_len=72, output_attention=0, dropout=0.1, d_model=144,n_heads=1, d_ff=128, activation='gelu', e_layers=1).cuda()

        if i == 0:
            model.load_state_dict(torch.load("/home/mw/project/best_model/best_test_72_itrans1_model_temp_1.pt"))
            model2.load_state_dict(torch.load("/home/mw/project/best_model/test_72_itrans1_model_temp_fold_1.pt"))
            model_ilstm.load_state_dict(torch.load("/home/mw/project/best_model/best_test_72_itrans2_model_temp_1.pt"))
        else:
            model.load_state_dict(torch.load("/home/mw/project/best_model/best_test_72_itrans1_model_wind_1.pt"))
            model2.load_state_dict(torch.load("/home/mw/project/best_model/test_72_itrans1_model_wind_fold_1.pt"))
            model_ilstm.load_state_dict(torch.load("/home/mw/project/best_model/best_test_72_itrans2_model_wind_1.pt"))

        model.eval()
        model2.eval()
        model_ilstm.eval()
        data1 = np.concatenate([covariate_1, data], axis=2)  # (N, L, C, S)
        data1 = data1.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data1 = data1.reshape(N * S, L, C)
        data1 = torch.tensor(data1).float().cuda()  # (N * S, L, C)
        outputs1 = model(data1)*0.6+model2(data1)*0.4
        outputs_ilstm1 = model_ilstm(data1)
        outputs1 = outputs1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        outputs_ilstm1 = outputs_ilstm1[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        del data1
        gc.collect()
        torch.cuda.empty_cache()

        data2 = np.concatenate([covariate_2, data], axis=2)  # (N, L, C, S)
        data2 = data2.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data2 = data2.reshape(N * S, L, C)
        data2 = torch.tensor(data2).float().cuda()  # (N * S, L, C)
        outputs2 = model(data2)*0.6+model2(data2)*0.4
        outputs_ilstm2 = model_ilstm(data2)
        outputs2 = outputs2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        outputs_ilstm2 = outputs_ilstm2[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        del data2
        gc.collect()
        torch.cuda.empty_cache()

        data3 = np.concatenate([covariate_3, data], axis=2)  # (N, L, C, S)
        data3 = data3.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data3 = data3.reshape(N * S, L, C)
        data3 = torch.tensor(data3).float().cuda()  # (N * S, L, C)
        outputs3 = model(data3)*0.6+model2(data3)*0.4
        outputs_ilstm3 = model_ilstm(data3)
        outputs3 = outputs3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        outputs_ilstm3 = outputs_ilstm3[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        del data3
        gc.collect()
        torch.cuda.empty_cache()

        data4 = np.concatenate([covariate_4, data], axis=2)  # (N, L, C, S)
        data4 = data4.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data4 = data4.reshape(N * S, L, C)
        data4 = torch.tensor(data4).float().cuda()  # (N * S, L, C)
        outputs4 = model(data4)*0.6+model2(data4)*0.4
        outputs_ilstm4 = model_ilstm(data4)
        outputs4 = outputs4[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        outputs_ilstm4 = outputs_ilstm4[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        del data4
        gc.collect()
        torch.cuda.empty_cache()


        P = outputs1.shape[1]
        forecast1 = outputs1.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_itrans1 = forecast1.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        forecast_ilstm1 = outputs_ilstm1.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_ilstm1 = forecast_ilstm1.transpose(0, 2, 1, 3)  # (N, P, S, 1)


        P = outputs2.shape[1]
        forecast2 = outputs2.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_itrans2 = forecast2.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        forecast_ilstm2 = outputs_ilstm2.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_ilstm2 = forecast_ilstm2.transpose(0, 2, 1, 3)  # (N, P, S, 1)


        P = outputs3.shape[1]
        forecast3 = outputs3.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_itrans3 = forecast3.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        forecast_ilstm3 = outputs_ilstm3.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_ilstm3 = forecast_ilstm3.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        P = outputs4.shape[1]
        forecast4 = outputs4.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_itrans4 = forecast4.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        forecast_ilstm4 = outputs_ilstm4.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast_ilstm4 = forecast_ilstm4.transpose(0, 2, 1, 3)  # (N, P, S, 1)

        if i==0:
            '''温度itrans大头'''
            forecast = (forecast_itrans1*0.45+fgg_temp_1*0.3+forecast_ilstm1*0.25+
                        forecast_itrans2*0.45+fgg_temp_2*0.3+forecast_ilstm2*0.25+
                        forecast_itrans3*0.45+fgg_temp_3*0.3+forecast_ilstm3*0.25+
                        forecast_itrans4*0.45+fgg_temp_4*0.3+forecast_ilstm4*0.25)/4
            forecast[:,:24]=forecast[:,:24]*0.5+chusai_temp_24*0.5
            forecast[:,24:48]=forecast[:,24:48]*0.5+chusai_temp_48[:,24:48]*0.5

        if i==1:
            '''风速soft大头'''
            forecast = (forecast_itrans1*0.3+fgg_wind_1*0.45+forecast_ilstm1*0.25+
                        forecast_itrans2*0.3+fgg_wind_2*0.45+forecast_ilstm2*0.25+
                        forecast_itrans3*0.3+fgg_wind_3*0.45+forecast_ilstm3*0.25+
                        forecast_itrans4*0.3+fgg_wind_4*0.45+forecast_ilstm4*0.25)/4
            forecast[:,:24]=forecast[:,:24]*0.5+chusai_wind_24*0.5
            forecast[:,24:48]=forecast[:,24:48]*0.5+chusai_wind_48[:,24:48]*0.5
            
        if i == 0:
            np.save(os.path.join(save_path, "temp_predict.npy"), forecast)
        else:
            np.save(os.path.join(save_path, "wind_predict.npy"), forecast)




