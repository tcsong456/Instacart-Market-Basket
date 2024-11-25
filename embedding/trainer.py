# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:06:40 2024

@author: congx
"""
import os
import gc
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from utils.utils import logger
from create_merged_data import data_processing
from utils.loss import NextBasketLoss,SeqLogLoss

get_checkpoint_path = lambda model_name:f'{model_name}_best_checkpoint.pth'
class Trainer:
    def __init__(self,
                 data,
                 prod_data,
                 output_dim,
                 learning_rate,
                 lagging=1,
                 optim_option='adam',
                 batch_size=32,
                 warm_start=False,
                 early_stopping=3,
                 epochs=100,
                 eval_epoch=1):
        optimizer_ = optim_option.lower()
        if optimizer_ == 'sgd':
            optimizer = optim.SGD
        elif optimizer_ == 'adam':
            optimizer = optim.Adam
        elif optimizer_ == 'adamw':
            optimizer = optim.AdamW
        else:
            logger.warning(f'{optimizer_} is an invalid option for optimizer')
            sys.exit(1)
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.eval_epoch = eval_epoch
        self.epochs = epochs
        self.output_dim=output_dim
        self.learning_rate=learning_rate
        self.data = data
        self.prod_data = prod_data
        self.lagging = lagging
        temp_list = ['order_dow','order_hour_of_day','time_zone','days_since_prior_order']
        self.temp_dim = sum([data[t].max()+1 for t in temp_list])
        self.max_len = data['order_number'].max()
        
        self.loss_fn_tr = SeqLogLoss(lagging=lagging,eps=1e-7)
        self.loss_fn_te = NextBasketLoss(lagging=lagging,eps=1e-7)
    
    def build_agg_data(self,attr_col):
        agg_data = data_processing(self.data,save=True)
        agg_data_tr,agg_data_te = agg_data[agg_data['eval_set']=='train'],agg_data[agg_data['eval_set']=='test']
        agg_data_tr = agg_data_tr[agg_data_tr[attr_col].map(lambda x:len(x)-self.lagging>=2)]
        agg_data_te = agg_data_te[agg_data_te[attr_col].map(lambda x:len(x)-(self.lagging+1)>=2)]
        agg_data = pd.concat([agg_data_tr,agg_data_te])
        del agg_data_tr,agg_data_te
        gc.collect()
        return agg_data
    
    def build_data_dl(self,mode):
        raise NotImplementedError('subclass must implement this method')
    
    def train(self,use_amp=False):
        model = self.model(self.input_dim,self.output_dim,*self.max_index_info).to('cuda')
        model_name = model.__class__.__name__
        optimizer = self.optimizer(model.parameters(),lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=0,factor=0.2,verbose=True)
        
        checkpoint_path = get_checkpoint_path(model_name)
        checkpoint_path = f'checkpoint/{checkpoint_path}'
        if self.warm_start:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['best_epoch']
            start_epoch += 1
            best_loss = checkpoint['best_loss']
            logger.info(f'warm starting training from best epoch:{start_epoch} and best loss:{best_loss:.5f}')
        else:
            start_epoch = 0
            best_loss = np.inf
        
        return model,optimizer,lr_scheduler,start_epoch,best_loss,checkpoint_path



#%%
