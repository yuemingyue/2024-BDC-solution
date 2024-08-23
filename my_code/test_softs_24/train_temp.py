import numpy as np
import os
import sys
from data_provider.data_loader import Dataset_Fuse
from models.SOFTS import SOFTS_MLP
import torch
import torch.nn as nn

path="/home/mw/input/bdc_train9198/global/"
root_path=path+'global'
data_path='temp.npy'
exter_path='wind.npy'
dataset=Dataset_Fuse(root_path,data_path,exter_path,[168,1,24],features='MS')



class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss,self).__init__()

    def forward(self,outputs,batch_y):
        return (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()


from torch.utils.data import DataLoader
import torch.nn as nn
import torch

#DATALOADER
bs=4096
shuffle_flag=True
loader=DataLoader(dataset,batch_size=bs,shuffle=shuffle_flag,num_workers=32)
#MODEL
device='cuda' if torch.cuda.is_available() else 'cpu'
enc_in,c_out,pred_len,device=37,1,24,device
model=SOFTS_MLP(seq_len=168,pred_len=24,d_model=96,dropout=.05,use_norm=True,d_core=64,d_ff=128,activation='gelu',e_layers=1).to(device)
#OPTIMIZER
lr=1e-2
wd=1e-5
optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
#LOSS
mseloss=nn.MSELoss()
loss_fn=FFTLoss()
#EPOCHS
epochs=2
#AMP
scaler = torch.cuda.amp.GradScaler()
use_amp=True
#FSNET
from tqdm import tqdm
bst_loss=9999999
for epo in range(epochs):
    epo_loss=[]
    for step,(batch_x,batch_y) in tqdm(enumerate(loader), total=len(loader), desc=f"Training epoch {epo+1}", leave=True):
        batch_x=batch_x.float().to(device)
        batch_y=batch_y.float().to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                #FORWARD
                batch_out=model(batch_x)
                batch_out = batch_out[:, -24:,-1:]
                batch_y = batch_y[:, -24:,-1:]
                mse=mseloss(batch_out,batch_y)
                loss=loss_fn(batch_out,batch_y)
        else:
            batch_out=model(batch_x)
            batch_out = batch_out[:, -24:,-1:]
            batch_y = batch_y[:, -24:,-1:]
            mse=mseloss(batch_out,batch_y)
            loss=loss_fn(batch_out,batch_y)
        #BACKWARD
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epo_loss.append(mse.item())
        if step%500==0 and step!=0:
            epo_loss_sum=sum(epo_loss)/len(epo_loss)
            print(f'now epoch{epo},now step{step}:mse_loss:{epo_loss_sum}')
            if epo_loss_sum<bst_loss:
                bst_loss=epo_loss_sum
                torch.save(model.state_dict(),'/home/mw/project/best_model/best_test_24_softs_model_temp.pt')
    epo_loss_sum=sum(epo_loss)/len(epo_loss)
    print(f'now epoch{epo}:mse_loss:{epo_loss_sum}')
    torch.save(model.state_dict(),f'/home/mw/project/best_model/test_24_softs_model_temp_epo_{epo}.pt')
use_amp=True