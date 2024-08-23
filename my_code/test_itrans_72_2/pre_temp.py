import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset,DataLoader
import os
# 导入temp dataset 
from data_provider.data_loader_temp import Dataset_Meteorology
# 导入模型
from models.itransformer_temp import itransformer
# ema 训练 动态学习率
from utils.tools import EMA
from utils.tools import get_linear_schedule_with_warmup

# 固定种子环境
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
seed_torch(3047)

path="/home/mw/input/bdc_train9198/global/"
root_path=path+'global'
data_path='temp.npy'
exter_path='wind.npy'
print('读取温度数据')
dataset=Dataset_Meteorology(root_path,data_path,exter_path,[168,1,72],features='MS') 
print('温度数据读取完成')



# 训练超参数设置
bs=7000
shuffle_flag=True
device='cuda' if torch.cuda.is_available() else 'cpu'
lr=0.009
epochs=3
kfold = 5


# 生成所有index   辅助完成k折训练
all_index = list(range(len(dataset)))

# k 折交叉
''' 根据线上效果固使用一折结果'''

skf = KFold(n_splits = kfold,random_state = None,shuffle = False)

for num_fold,(train_idx,valid_idx) in enumerate(skf.split(all_index,all_index)):
    print(f"***********************************fold:{num_fold+1}数据加载**************************************")
    train_set = Subset(dataset,train_idx)
    # valid_set = Subset(dataset,valid_idx)  # 注释因为无需验证
    train_loader = DataLoader(train_set,batch_size=bs,shuffle=shuffle_flag,num_workers=8)
    # valid_loader= DataLoader(valid_set,batch_size=bs,shuffle=shuffle_flag,num_workers=8)
    print(f"**********************************fold:{num_fold+1}模型加载*****************************************")
    # 初始化模型 配置
    model = itransformer(seq_len=168,pred_len=72,output_attention=0,dropout=0.2,d_model=64,n_heads=1,d_ff=128,activation='gelu',e_layers=1).to(device)
    #AMP
    scaler = torch.cuda.amp.GradScaler()
    use_amp=False
    loss_mode='mix'
    
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
            # print(batch_x.shape)
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
            regularization_strength = 3e-5  # 这个系数可以根据需要调整
            
            if loss_mode=='mse':
                loss = loss_fn(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode=='feq':
                loss = (torch.fft.rfft(batch_out, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()*0.5+0.5*loss_fn(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode=='mix':
                loss = loss_fn(batch_out,batch_y)*0.5+0.5*loss_fn_mae(batch_out,batch_y)+regularization_strength * l1_penalty
            elif loss_mode == 'diff':
                loss = loss_fn(batch_out,batch_y)+DiffLoss(batch_out,batch_y)
                
            #BACKWARD
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update()
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
                print(f'now fold {num_fold+1} epoch{epo+1},now step{step},loss:{loss_sum},mse:{mse_sum},score:{score_sum}' )
                if score_sum<best_score:
                    print("save_model:",score_sum)
                    best_score=score_sum
                    torch.save(model.state_dict(),f'/home/mw/project/best_model/best_test_72_itrans2_model_temp_{num_fold+1}.pt')
                    

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

    break



