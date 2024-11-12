# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:36:46 2024

@author: congx
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

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
                max_len,
                num_layers=1,
                batch_first=True,
                dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            output_dim,
                            batch_first=batch_first,
                            num_layers=num_layers,
                            dropout=dropout)
        self.user_embedding = nn.Embedding(max_users,emb_dim)
        self.product_embedding = nn.Embedding(max_products,emb_dim)
        self.aisle_embedding = nn.Embedding(max_aisles,emb_dim)
        self.dept_embedding = nn.Embedding(max_depts,emb_dim)
        self.max_len = max_len
        
        hidden_dim = output_dim // 2
        self.h = nn.Linear(output_dim,hidden_dim)
        self.final = nn.Linear(hidden_dim,1)
    
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
        h = self.h(outputs)
        final_results = self.final(h).squeeze()
        batch_len,max_seq = final_results.shape
        if max_seq < self.max_len:
            padded_length = self.max_len - max_seq
            padded_zeros = torch.zeros(batch_len,padded_length).to('cuda')
            final_results = torch.cat([final_results,padded_zeros],dim=1)
        return final_results
        


#%%
