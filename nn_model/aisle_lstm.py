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
                 max_user,
                 max_aisle,
                 max_dept,
                 max_len
                 ):
        super().__init__()
        self.user_embedding = nn.Embedding(max_user,50)
        self.aisle_embedding = nn.Embedding(max_aisle,50)
        self.dept_embedding = nn.Embedding(max_dept,50)
        self.max_len = max_len
        
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=True)
        self.final = nn.Linear(output_dim,1)
        self.input_dim = input_dim
    
    def forward(self,x,users,aisles,depts,dows,hours,tzs,days):
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        oh_tzs = F.one_hot(tzs,num_classes=28)
        oh_days = F.one_hot(days,num_classes=31)

        temp = torch.cat([oh_dows,oh_hours,oh_tzs,oh_days],dim=-1)
        user_embedding = self.user_embedding(users)
        aisle_embedding = self.aisle_embedding(aisles)
        dept_embedding = self.dept_embedding(depts)
        embedding = torch.cat([user_embedding,aisle_embedding,dept_embedding],dim=-1)
        embedding = embedding.unsqueeze(1).repeat(1, self.max_len, 1) 
        x = torch.cat([x,temp,embedding],dim=-1)

        outputs,_ = self.lstm(x)
        h = self.final(outputs)
        outputs = torch.cat([outputs,torch.sigmoid(h)],dim=-1)
        h = h.squeeze()
        return outputs,h

#%%
