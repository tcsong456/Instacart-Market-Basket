# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:46:42 2024

@author: congx
"""
import torch
from torch import nn

class BPRLoss(nn.Module):
    def __init__(self,
                 lambda_reg):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self,v_i,v_k,v_j):
        positive_score = (v_i * v_k).sum(dim=1).reshape(-1,1)
        negative_score = (v_k.unsqueeze(1) * v_j).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(positive_score - negative_score)).mean()
        reg_loss = self.lambda_reg * (v_i.norm(2) + v_k.norm(2) + v_j.norm(2))
        loss += reg_loss
        return loss

class SeqLogLoss(nn.Module):
    def __init__(self,
                 lagging,
                 eps=1e-7):
        super().__init__()
        self.eps = eps
        self.lagging=lagging
        
    def forward(self,y_pred,y_true,lengths):
        y_pred = torch.sigmoid(y_pred)
        logloss = y_true * torch.log(y_pred + self.eps) + (1 - y_true) * torch.log(1 - y_pred + self.eps)
        mask = torch.zeros(y_true.shape).cuda()
        for idx,length in enumerate(lengths):
            mask[idx,:length-self.lagging] = 1
        
        total = sum(lengths) - self.lagging * len(lengths)
        loss = -(logloss * mask).sum() / total
        return loss
    
class NextBasketLogLoss(SeqLogLoss):
    def __init__(self,
                 lagging,
                 eps=1e-7):
        super().__init__(eps=eps,
                         lagging=lagging)
    
    def forward(self,y_pred,y_true,lengths):
        y_pred = torch.sigmoid(y_pred)
        index = torch.Tensor(lengths).long().reshape(-1,1).to('cuda') - self.lagging
        yp = torch.gather(y_pred,dim=1,index=index).squeeze()
        yt = torch.gather(y_true,dim=1,index=index).squeeze()
        loss = yt * torch.log(yp + self.eps) + (1 - yt) * torch.log(1 - yp + self.eps)
        loss = -loss.sum() / y_pred.shape[0]
        return loss

class SeqMSELoss(nn.Module):
    def __init__(self,
                 lagging):
        super().__init__()
        self.lagging = lagging
    
    def forward(self,y_pred,y_true,lengths):
        mask = torch.zeros(y_true.shape).cuda()
        for idx,length in enumerate(lengths):
            mask[idx,:length-self.lagging] = 1
        
        mseloss = (y_true - y_pred)**2
        total = sum(lengths) - self.lagging * len(lengths)
        loss = torch.sum(mseloss * mask) / total
        return loss

class NextBasketMSELoss(nn.Module):
    def __init__(self,
                 lagging):
        super().__init__()
        self.lagging = lagging
    
    def forward(self,y_pred,y_true,lengths):
        index = torch.Tensor(lengths).long().reshape(-1,1).to('cuda') - self.lagging
        yp = torch.gather(y_pred,dim=1,index=index).squeeze()
        yt = torch.gather(y_true,dim=1,index=index).squeeze()
        loss = (yt - yp)**2
        loss = loss.sum() / y_pred.shape[0]
        return loss
    
        
        #%%


