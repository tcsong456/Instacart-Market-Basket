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

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()
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
    
    def _collect_final_time_step(self,lengths,aux_info,time_steps,label=None):
        batch_lengths = torch.Tensor(lengths).reshape(-1,1).unsqueeze(-1).expand(-1,1,time_steps.shape[-1]).long().cuda() - 1
        pred_emb = torch.gather(time_steps,index=batch_lengths,dim=1).squeeze()
        pred_emb = pred_emb.cpu().numpy()
        user_attr = torch.stack(aux_info[:2],dim=1).cpu().numpy() + 1
        pred_emb = np.concatenate([user_attr,pred_emb],axis=1)
        if label is not None:
            index = torch.Tensor(lengths).reshape(-1,1).long().cuda() - 1
            label = torch.gather(label,dim=1,index=index)
            label = label.cpu().numpy()
            pred_emb = np.concatenate([pred_emb,label],axis=1)
        
        return pred_emb
    
    def predict(self,save_name):
        predict_data = self.agg_data[self.agg_data['eval_set']=='test']
        data_te = self.data_maker(predict_data,self.max_len,*self.data_maker_dicts,mode='test')
        users,feat_dict,temporal_dict,feat_dim = data_te
        for key,value in temporal_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temporal_dict[key] = value
        model = self.model(self.input_dim,self.output_dim,*self.max_index_info).to('cuda')
        checkpoint = torch.load(self.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_dl = self.dataloader(users,feat_dict,temporal_dict,1024,False,False)
        test_batch_loader = tqdm(test_dl,total=users.shape[0]//1024,desc='predicting next reorder basket',
                                 leave=False,dynamic_ncols=True)
        predictions = []
        with torch.no_grad():
            for batch,batch_lengths,temps,*aux_info in test_batch_loader:
                batch = batch[:,:,:-1]
                batch = torch.from_numpy(batch).cuda()
                temps = [torch.stack(temp) for temp in temps]
                aux_info = [convert_index_cuda(b) for b in aux_info]
                
                h,preds = model(batch,*aux_info,*temps)
                # preds = torch.sigmoid(preds).cpu()
                # index = torch.Tensor(batch_lengths).long().reshape(-1,1) - 1
                # probs = torch.gather(preds,dim=1,index=index).numpy()
                
                # users,attrs = aux_info[:2]
                # users += 1;attrs += retore_attr_extent
                # user_attr = np.stack([users,attrs],axis=1)
                # user_attr_prob  = np.concatenate([user_attr,probs],axis=1)
                # predictions.append(user_attr_prob)
                pred_emb_te = self._collect_final_time_step(batch_lengths,aux_info,h)
                predictions.append(pred_emb_te)
        
        predictions = np.concatenate(predictions).astype(np.float32)
        os.makedirs('metadata',exist_ok=True)
        pred_path = f'metadata/{save_name}.npy'
        np.save(pred_path,predictions)
        logger.info(f'predictions saved to {pred_path}')
        return predictions




#%%
