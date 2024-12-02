# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:34:02 2024

@author: congx
"""
import torch
from torch import nn

class BPREmbeddingModel(nn.Module):
    def __init__(self,
                 num_items,
                 emb_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items,emb_dim)
    
    def forward(self,items):
        item_target,item_pos,item_neg = items[:,0],items[:,1],items[:,2:]
        v_i = self.item_embedding(item_target)
        v_k = self.item_embedding(item_pos)
        v_j = self.item_embedding(item_neg)
        return v_i,v_k,v_j


#%%
