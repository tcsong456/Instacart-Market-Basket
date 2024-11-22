# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:36:46 2024

@author: congx
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
convert_to_tensor = lambda x:torch.Tensor(x).to('cuda')

class ProdLSTM(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                max_users,
                max_products,
                max_aisles,
                max_depts,
                max_len):
        super().__init__()
        emb_dim = 50
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=True)
        self.user_embedding = nn.Embedding(max_users,emb_dim)
        self.product_embedding = nn.Embedding(max_products,emb_dim)
        self.aisle_embedding = nn.Embedding(max_aisles,emb_dim)
        self.dept_embedding = nn.Embedding(max_depts,emb_dim)
        self.max_len = max_len
        self.final = nn.Linear(output_dim,1)
    
    def forward(self,inputs,users,products,aisles,depts,dows,hours,tzs):
        batch,seq_len,_ = inputs.shape

        oh_tzs = F.one_hot(tzs,num_classes=28)
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        tmp = torch.cat([oh_dows,oh_hours,oh_tzs],dim=-1)

        user_embedding = self.user_embedding(users)
        product_embedding = self.product_embedding(products)
        aisle_embedding = self.aisle_embedding(aisles)
        dept_embedding = self.dept_embedding(depts)
        embedding = torch.cat([user_embedding,product_embedding,aisle_embedding,dept_embedding],dim=-1)
        embedding = embedding.unsqueeze(1).repeat(1, self.max_len, 1) 
        inputs = torch.cat([inputs,tmp,embedding],dim=-1)
        
        outputs,_ = self.lstm(inputs)

        final_results = self.final(outputs).squeeze()
        return final_results

class ProdLSTMV1(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                max_users,
                max_products,
                max_aisles,
                max_depts,
                max_len,
                aisle_dict):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=True)
        emb_dim=50
        self.aisle_dict = aisle_dict
        self.user_embedding = nn.Embedding(max_users,emb_dim)
        self.product_embedding = nn.Embedding(max_products,emb_dim)
        self.aisle_embedding = nn.Embedding(max_aisles,21)
        self.dept_embedding = nn.Embedding(max_depts,emb_dim)
        self.max_len = max_len
        self.final = nn.Linear(output_dim,1)
    
    def forward(self,inputs,users,products,aisles,depts,dows,hours,tzs):
        batch,seq_len,_ = inputs.shape

        oh_tzs = F.one_hot(tzs,num_classes=28)
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        tmp = torch.cat([oh_dows,oh_hours,oh_tzs],dim=-1)

        user_embedding = self.user_embedding(users)
        product_embedding = self.product_embedding(products)
        dept_embedding = self.dept_embedding(depts)
        embedding = torch.cat([user_embedding,product_embedding,dept_embedding],dim=-1)
        embedding = embedding.unsqueeze(1).repeat(1, self.max_len, 1) 
        
        aisle_params = []
        users,aisles = users.cpu().numpy(),aisles.cpu().numpy()
        users += 1;aisles += 1
        keys = np.stack([users,aisles],axis=1)
        keys = list(map(tuple,map(np.squeeze,np.split(keys,keys.shape[0],axis=0))))
        # aisle_params = list(map(self.aisle_dict.get,keys))
        aisle_params = list(map(lambda key:self.aisle_dict.get(key,self.aisle_embedding(torch.Tensor([key[1]-1]).long().cuda()\
                                                               ).repeat(self.max_len,1).cpu()),keys))
        aisle_params = np.stack(aisle_params)
        aisle_params = torch.from_numpy(aisle_params).cuda()
        
        inputs = torch.cat([inputs,tmp,aisle_params,embedding],dim=-1)
        outputs,_ = self.lstm(inputs)

        final_results = self.final(outputs).squeeze()
        return final_results

class TemporalNet(ProdLSTM):
    def __init__(self,
                 input_dim,
                 output_dim,
                 emb_dim,
                 max_users,
                 max_products,
                 max_aisles,
                 max_depts,
                 max_len,
                 skip_channels,
                 residual_channels,
                 kernel_sizes=[2]*4,
                 dilations=[2**i for i in range(4)]):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         emb_dim=emb_dim,
                         max_users=max_users,
                         max_products=max_products,
                         max_aisles=max_aisles,
                         max_depts=max_depts,
                         max_len=max_len)
    
        self.conv_lists = nn.ModuleList()
        self.linear_lists = nn.ModuleList()
        for kernel_size,dilation in zip(kernel_sizes,dilations):
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(residual_channels,
                             2*residual_channels,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             padding=padding)
            self.linear_lists.append(nn.Linear(residual_channels,skip_channels+residual_channels))
            self.conv_lists.append(conv)
            
        self.init_linear = nn.Linear(input_dim,residual_channels)
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.final = nn.Linear(skip_channels*len(kernel_sizes)+output_dim,1)
    
    def forward(self,inputs,lengths,users,products,aisles,depts,dows,hours,tzs):
        lstm_temp,x = super().forward(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
        
        skip_outputs = []
        input_seq = x.shape[1]
        x = self.init_linear(x)
        for conv,linear in zip(self.conv_lists,self.linear_lists):
            inputs = x.transpose(1,2)
            inputs = conv(inputs)
            inputs = inputs[:,:,:input_seq]
            inputs = inputs.transpose(1,2)
            conv_filter,conv_gate = torch.split(inputs,self.residual_channels,dim=-1)
            dilation = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
            output = linear(dilation)
            skip,residual = torch.split(output,[self.skip_channels,self.residual_channels],dim=-1)
            x = x + residual
            skip_outputs.append(skip)
        skip_outputs = torch.cat(skip_outputs,dim=-1)
        final_results = torch.cat([lstm_temp,skip_outputs],dim=-1)
        preds = self.final(final_results).squeeze()
        return preds


#%%
# from torch import nn
# x = {(1,2):torch.rand(20,24),(3,4):torch.rand(20,24)}
# y = nn.Embedding(50,24)
# c = list(map(lambda key:x.get(key,y(torch.Tensor([key[1]]).long()).repeat(20,1)),[(1,2),(3,4),(5,6)]))
# torch.stack(c).shape



