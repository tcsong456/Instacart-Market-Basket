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
from torch.cuda.amp import autocast,GradScaler
from create_merged_data import data_processing
from utils.loss import NextBasketLogLoss,SeqLogLoss

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
        self.loss_fn_te = NextBasketLogLoss(lagging=lagging,eps=1e-7)
    
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

    def _collect_final_time_step(self,lengths,aux_info,time_steps,label=None):
        batch_lengths = torch.Tensor(lengths).reshape(-1,1).unsqueeze(-1).expand(-1,1,time_steps.shape[-1]).long().cuda() - self.lagging
        pred_emb = torch.gather(time_steps,index=batch_lengths,dim=1).squeeze()
        pred_emb = pred_emb.cpu().numpy()
        user_attr = torch.stack(aux_info[:2],dim=1).cpu().numpy() + 1
        pred_emb = np.concatenate([user_attr,pred_emb],axis=1)
        if label is not None:
            index = torch.Tensor(lengths).reshape(-1,1).long().cuda() - self.lagging
            label = torch.gather(label,dim=1,index=index)
            label = label.cpu().numpy()
            pred_emb = np.concatenate([pred_emb,label],axis=1)
        
        return pred_emb
    
    def train(self,use_amp=False):
        core_info_tr = self.build_data_dl(mode='train')
        if 'product' in self.model_name.lower():
            model = self.model(self.input_dim,self.output_dim,*self.max_index_info,32,32,[2]*5,[2**i for i in range(5)])
        else:
            model = self.model(self.input_dim,self.output_dim,*self.max_index_info)
        model = model.to('cuda')
        optimizer = self.optimizer(model.parameters(),lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=0,factor=0.2,verbose=True)
        checkpoint_path = get_checkpoint_path(self.model_name)
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
        self.checkpoint_path = checkpoint_path
            
        no_improvement = 0
        for epoch in range(start_epoch,self.epochs):
            total_loss,cur_iter = 0,1
            model.train()
            train_dl = self.dataloader(*core_info_tr,self.batch_size,True,True)
            train_batch_loader = tqdm(train_dl,total=core_info_tr[0].shape[0]//self.batch_size,desc=f'training next {self.attr} basket at epoch:{epoch}',dynamic_ncols=True,leave=False)
            
            for batch,batch_lengths,temps,*aux_info in train_batch_loader:
                batch,label = batch[:,:,:-1],batch[:,:,-1]
                label = torch.from_numpy(label).to('cuda')
                batch = torch.from_numpy(batch).cuda()
                temps = [torch.stack(temp) for temp in temps]
                aux_info = [convert_index_cuda(b) for b in aux_info]
                
                optimizer.zero_grad()
                if use_amp:
                    scaler = GradScaler()
                    with autocast():
                        h,preds = model(batch,*temps)
                        loss = self.loss_fn_tr(preds,label,batch_lengths)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    h,preds = model(batch,*aux_info,*temps)
                    loss = self.loss_fn_tr(preds,label,batch_lengths)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                cur_loss = loss.item()
                total_loss += cur_loss
                avg_loss = total_loss / cur_iter
                    
                cur_iter += 1
                train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
            
            if epoch % self.eval_epoch == 0:
                pred_embs_eval = []
                model.eval()
                total_loss_val,ite = 0,1
                eval_dl = self.dataloader(*core_info_tr,1024,False,False)
                eval_batch_loader = tqdm(eval_dl,total=core_info_tr[0].shape[0]//1024,desc=f'evaluating next {self.attr} basket',
                                          leave=False,dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch,batch_lengths,temps,*aux_info in eval_batch_loader:
                        batch,label = batch[:,:,:-1],batch[:,:,-1]
                        batch = torch.from_numpy(batch).cuda()
                        label = torch.from_numpy(label).to('cuda')
                        temps = [torch.stack(temp) for temp in temps]
                        aux_info = [convert_index_cuda(b) for b in aux_info]
                        
                        h,preds = model(batch,*aux_info,*temps)
                        loss = self.loss_fn_te(preds,label,batch_lengths)
                        
                        total_loss_val += loss.item()
                        avg_loss_val = total_loss_val / ite
                        ite += 1
                        eval_batch_loader.set_postfix(eval_loss=f'{avg_loss_val:.05f}')
                        
                        pred_emb_val = self._collect_final_time_step(batch_lengths,aux_info,h,label)
                        pred_embs_eval.append(pred_emb_val)
                lr_scheduler.step(avg_loss_val)
                pred_embs_eval = np.concatenate(pred_embs_eval).astype(np.float32)
                
                if avg_loss_val < best_loss:
                    best_loss = avg_loss_val
                    checkpoint = {'best_epoch':epoch,
                                  'best_loss':best_loss,
                                  'model_state_dict':model.state_dict(),
                                  'optimizer_state_dict':optimizer.state_dict()}
                    os.makedirs('checkpoint',exist_ok=True)
                    torch.save(checkpoint,checkpoint_path)
                    np.save(f'metadata/user_{self.attr}_eval.npy',pred_embs_eval)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                if no_improvement == self.early_stopping:
                    logger.info('early stopping is trggered,the model has stopped improving')
                    return
    
    def predict(self,save_name):
        predict_data = self.agg_data[self.agg_data['eval_set']=='test']
        data_te = self.data_maker(predict_data,self.max_len,*self.data_maker_dicts,mode='test')
        users,feat_dict,temporal_dict,feat_dim = data_te
        for key,value in temporal_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temporal_dict[key] = value
        if 'product' in self.model_name.lower():
            model = self.model(self.input_dim,self.output_dim,*self.max_index_info,32,32,[2]*5,[2**i for i in range(5)])
        else:
            model = self.model(self.input_dim,self.output_dim,*self.max_index_info)
        model = model.to('cuda')
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
                pred_emb_te = self._collect_final_time_step(batch_lengths,aux_info,h)
                predictions.append(pred_emb_te)
        
        predictions = np.concatenate(predictions).astype(np.float32)
        os.makedirs('metadata',exist_ok=True)
        pred_path = f'metadata/{save_name}.npy'
        np.save(pred_path,predictions)
        logger.info(f'predictions saved to {pred_path}')
        return predictions




#%%


