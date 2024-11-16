# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:36:46 2024

@author: congx
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
torch.autograd.set_detect_anomaly(True)
convert_to_tensor = lambda x:torch.Tensor(x).to('cuda')

class ProdLSTM(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                emb_dim,
                max_users,
                max_products,
                max_aisles,
                max_depts,
                max_len):
        super().__init__()
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
        # return outputs,inputs

        final_results = self.final(outputs).squeeze()
        return final_results

class ProdWavnet(nn.Module):
    def __init__(self,
                 emb_dim,
                 max_users,
                 max_products,
                 max_aisles,
                 max_depts,
                 max_len,
                 in_channels,
                 skip_channels,
                 residual_channels,
                 kernel_sizes=[2]*4,
                 dilations=[2**i for i in range(4)]):
        super().__init__()
        self.user_embedding = nn.Embedding(max_users,emb_dim)
        self.product_embedding = nn.Embedding(max_products,emb_dim)
        self.aisle_embedding = nn.Embedding(max_aisles,emb_dim)
        self.dept_embedding = nn.Embedding(max_depts,emb_dim)
        
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
        self.init_linear = nn.Linear(in_channels,residual_channels)
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.final = nn.Linear(skip_channels*len(kernel_sizes),1)
        self.max_len = max_len
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
    
    def forward(self,x,lengths,users,products,aisles,depts,dows,hours,tzs):
        batch,seq_len,_ = x.shape
        oh_tzs = F.one_hot(tzs,num_classes=28)
        oh_dows = F.one_hot(dows,num_classes=7)
        oh_hours = F.one_hot(hours,num_classes=24)
        tmp = torch.cat([oh_dows,oh_hours,oh_tzs],dim=-1)
        user_embedding = self.user_embedding(users)
        product_embedding = self.product_embedding(products)
        aisle_embedding = self.aisle_embedding(aisles)
        dept_embedding = self.dept_embedding(depts)
        embedding = torch.cat([user_embedding,product_embedding,aisle_embedding,dept_embedding],dim=-1)
        embedding = embedding.repeat(1,self.max_len).reshape(batch,self.max_len,-1)
        # x[:,:,-1] = x[:,:,-1] / 30
        x = torch.cat([x,tmp,embedding],dim=-1)
        max_len = max(lengths)
        x = x[:,:max_len,:]
        
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
        final_results = self.final(skip_outputs).squeeze()
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
        max_len = max(lengths)
        
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
# from utils.utils import Timer
# import torch.profiler as profiler

# inputs = torch.rand(32,100,9).cuda()
# lengths = list(np.random.randint(2,81,(32,)))
# users = torch.randint(1,200000,(32,)).cuda()
# products = torch.randint(1,45000,(32,)).cuda()
# aisles = torch.randint(1,50,(32,)).cuda()
# depts = torch.randint(1,20,(32,)).cuda()
# dows = torch.randint(0,7,(32,100)).cuda()
# hours = torch.randint(0,24,(32,100)).cuda()
# tzs = torch.randint(0,28,(32,100)).cuda()
# temp_model = TemporalNet(input_dim=268,
#                     output_dim=100,
#                     emb_dim=50,
#                     max_users=200000,
#                     max_products=50000,
#                     max_aisles=50,
#                     max_depts=20,
#                     max_len=100,
#                     skip_channels=32,
#                     residual_channels=32,
#                     kernel_sizes=[2]*4,
#                     dilations=[2**i for i in range(4)]).cuda()
# lstm_model = ProdLSTM(input_dim=268,
#                     output_dim=100,
#                     emb_dim=50,
#                     max_users=200000,
#                     max_products=50000,
#                     max_aisles=50,
#                     max_depts=20,
#                     max_len=100).cuda()
# wavent_model = ProdWavnet(emb_dim=50,
#                         max_users=200000,
#                         max_products=50000,
#                         max_aisles=50,
#                         max_depts=20,
#                         max_len=100,
#                         in_channels=268,
#                         skip_channels=32,
#                         residual_channels=32,
#                         kernel_sizes=[2]*4,
#                         dilations=[2**i for i in range(4)]).cuda()
# with Timer(5):
#     # for _ in range(20):
#         # x = wavent_model(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
#         # y = lstm_model(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
#     with profiler.profile(
#                         activities=[
#                             profiler.ProfilerActivity.CPU,  # Profile CPU activities
#                             profiler.ProfilerActivity.CUDA  # Profile GPU activities
#                         ],
#                     ) as prof:
#         # y = lstm_model(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
#         w = temp_model(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
#         # x = wavent_model(inputs,lengths,users,products,aisles,depts,dows,hours,tzs)
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


#%%

    


