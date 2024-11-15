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
    
    def forward(self,inputs,lengths,users,products,aisles,depts,dows,hours,tzs):
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
        embedding = embedding.repeat(1,self.max_len).reshape(batch,self.max_len,-1)
        inputs[:,:,-1] = inputs[:,:,-1] / 30
        inputs = torch.cat([inputs,tmp,embedding],dim=-1)
        
        packed_inputs = rnn_utils.pack_padded_sequence(inputs,lengths,batch_first=True,enforce_sorted=False)
        packed_outputs,_ = self.lstm(packed_inputs)
        outputs,_ = rnn_utils.pad_packed_sequence(packed_outputs,batch_first=True)
        return outputs

        # final_results = self.final(outputs).squeeze()
        # return final_results

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
        x[:,:,-1] = x[:,:,-1] / 30
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
        super().__init__(emb_dim=emb_dim,
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
        self.init_linear = nn.Linear(in_channels,residual_channels)
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.final = nn.Linear(skip_channels*len(kernel_sizes),1)
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

#%%
# torch.cat([torch.rand(32,10,5),torch.zeros(32,0,5)],dim=1).shape
# checkpoint = torch.load('checkpoint/ProdWavnet_best_checkpoint.pth')
from utils.utils import Timer
import torch
import torch.nn as nn
import torch.nn.functional as F

