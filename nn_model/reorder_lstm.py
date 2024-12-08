# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:13:23 2024

@author: congx
"""

import torch
from torch import nn
from torch.nn import functional as F

class ReorderLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 max_user,
                 max_len
                 ):
        super().__init__()
        self.user_embedding = nn.Embedding(max_user,50)
        self.max_len = max_len
        
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=True)
        self.final = nn.Linear(output_dim,1)
        self.input_dim = input_dim
    
    def forward(self,x,users,dows,hours,tzs,days):
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        oh_tzs = F.one_hot(tzs,num_classes=28)
        oh_days = F.one_hot(days,num_classes=31)
        temp = torch.cat([oh_dows,oh_hours,oh_tzs,oh_days],dim=-1)
        
        user_embedding = self.user_embedding(users)
        embedding = user_embedding.unsqueeze(1).repeat(1, self.max_len, 1) 
        x = torch.cat([x,temp,embedding],dim=-1)

        outputs,_ = self.lstm(x)
        h = self.final(outputs)
        outputs = torch.cat([outputs,torch.expm1(h)],dim=-1)
        h = h.squeeze()
        return outputs,h
