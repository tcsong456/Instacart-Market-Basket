# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:52:54 2024

@author: congx
"""
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 batch_size=32,
                 epochs=200,
                 learning_rate=0.0005,
                 optimizer='adam',
                 reg_const=0.0,
                 collate_fn=None,
                 loss='log_loss'
                 ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.reg_const = reg_const
        self.loss = loss
        self.learning_rate = learning_rate
        self.train_data_loader = DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate_fn)
    
    # def calculate_loss(self):
    #     if self.loss == 'log_loss':
    #         return nn.BCELoss(reduction='mean')
    
    def calculate_loss(self,y_pred,y_target,pos_weight=100.0):
        loss_fn = nn.BCELoss(reduction='none')
        loss = loss_fn(y_pred,y_target)
        weight = y_target * pos_weight + (1 - y_target)
        return torch.mean(weight * loss)

    # def calculate_loss(self,y_pred,y_true, gamma=4.0, alpha=0.25):
    #     loss_fn = nn.BCELoss(reduction='none')
    #     bce = loss_fn(y_pred,y_true)
    #     pt = torch.exp(-bce)  # probability of the true class
    #     focal_loss = alpha * (1 - pt)**gamma * bce
    #     return torch.mean(focal_loss)

    
    def loss_fn(self,y_pred,y_target):
        y_target = y_target.to(torch.bool)
        y = y_pred[y_target]
        loss = torch.log(y+1e-5).sum()
        num = y_target.sum()
        return -loss / num
    
    def get_optimizer(self):
        if self.optimizer == 'sgd':
            return optim.SGD
        elif self.optimizer == 'adam':
            return optim.Adam
        elif self.optimizer == 'adamw':
            return optim.AdamW
    
    def fit(self):
        self.model.train()
        # loss_fn = self.calculate_loss()
        optimizer = self.get_optimizer()
        optimizer = optimizer(self.model.parameters(),lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            total_loss = 0
            batch_num = 1
            # try:
            with tqdm(self.train_data_loader,desc=f'epoch:{epoch}') as pbar:
                for batch in pbar:
                    preds = self.model(*batch[:-1])
                    targets = batch[-1].to(torch.float32).to('cuda')
                    
                    optimizer.zero_grad()
                    loss = self.calculate_loss(preds,targets)
                    loss_show = self.loss_fn(preds,targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss_show.item()
                    pbar.set_postfix({'batch_loss':total_loss/batch_num})
                    batch_num += 1
            # except RuntimeError:
            #     pass
                    



#%%


