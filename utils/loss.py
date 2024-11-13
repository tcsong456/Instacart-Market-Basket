# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:46:42 2024

@author: congx
"""
import torch
from torch import nn

class SeqLogLoss(nn.Module):
    def __init__(self,
                 eps=1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self,y_pred,y_true,lengths):
        max_len = max(lengths)
        y_true = y_true[:,:max_len]
        y_pred = torch.sigmoid(y_pred)
        logloss = y_true * torch.log(y_pred + self.eps) + (1 - y_true) * torch.log(1 - y_pred + self.eps)
        mask = torch.zeros(y_true.shape).cuda()
        for idx,length in enumerate(lengths):
            mask[idx,:length-1] = 1
        
        total = sum(lengths) - len(lengths)
        loss = -(logloss * mask).sum() / total
        return loss
    
class NextBasketLoss(SeqLogLoss):
    def __init__(self,
                 eps=1e-7):
        super().__init__()
    
    def forward(self,y_pred,y_true,lengths):
        y_pred = torch.sigmoid(y_pred)
        index = torch.Tensor(lengths).long().reshape(-1,1).to('cuda') - 1
        yp = torch.gather(y_pred,dim=1,index=index).squeeze()
        loss = y_true * torch.log(yp + self.eps) + (1 - y_true) * torch.log(1 - yp + self.eps)
        loss = -loss.sum() / y_pred.shape[0]
        return loss
        
        
        #%%
