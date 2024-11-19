# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:36:22 2024

@author: congx
"""
import torch
from torch import nn
from torch.nn import functional as F

class AisleLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=True)
        self.final = nn.Linear(output_dim,1)
    
    def forward(self,x1,x2,dows,hours,tzs):
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        oh_tzs = F.one_hot(tzs,num_classes=28)
        temp = torch.cat([oh_dows,oh_hours,oh_tzs],dim=-1)
        x = torch.cat([x1,x2,temp],dim=-1).to(torch.float32)
        outputs,_ = self.lstm(x)
        h = self.final(outputs).squeeze()
        return h

#%%
model = AisleLSTM(72,100).cuda()
temp = batch[-3]
temp = [torch.from_numpy(np.stack(b)).long().cuda() for b in temp]
x = model(torch.from_numpy(batch[0][:,:,:-1]).cuda(),torch.from_numpy(batch[2][:,:,:-1]).cuda(),*temp)

#%%
x.shape