# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:50:57 2024

@author: congx
"""
import gc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from utils.utils import pad,logger,TMP_PATH,pickle_save_load,optimize_dtypes
from torch.cuda.amp import autocast,GradScaler
from nn_model.aisle_lstm import AisleLSTM
from embedding.trainer import Trainer
from more_itertools import unique_everseen
from embedding.base_stats_collector import BaseStatsCollector

def aisle_data_maker(agg_data,data,max_len,aisle_dept_dict,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_aisle','temporal_dict','aisle_data_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'aisle_data_dict_{suffix}'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        aisle_info_dim = data_dict[rand_key][0].shape[-1] - 1
        user_aisle = pickle_save_load(os.path.join(TMP_PATH,f'user_aisle_{suffix}'),mode='load')
        temp_data = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{suffix}'),mode='load')
        return user_aisle,data_dict,temp_data,aisle_info_dim
    
    user_aisle = []
    temporal_dict,data_dict = {},{}
    dpar = tqdm(agg_data.iterrows(),total=len(agg_data),desc='building aisle data for dataloader',
                dynamic_ncols=True,leave=False)
    for _,row in dpar:
        user_id,aisles,reorders = row['user_id'],row['aisle_id'],row['reordered']
        temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
        if mode != 'test' and row['eval_set'] == 'test':
            aisles = aisles[:-1]
            reorders = reorders[:-1]
            for idx,col_data in enumerate(temporal_cols):
                col_data = col_data[:-1]
                temporal_cols[idx] = col_data
        order_dow,order_hour,order_tz,order_days = temporal_cols
        order_dow = pad(np.roll(order_dow,-1)[:-1],max_len)
        order_hour = pad(np.roll(order_hour,-1)[:-1],max_len)
        order_tz = pad(np.roll(order_tz,-1)[:-1],max_len)
        order_days = np.roll(order_days,-1)[:-1]
        temporal_dict[user_id] = [order_dow,order_hour,order_tz]
        
        aisles,next_aisles = aisles[:-1],aisles[-1]
        next_aisles = next_aisles.split('_')
        orders = [aisle.split('_') for aisle in aisles]
        all_aisles = list(set(chain.from_iterable([aisle.split('_') for aisle in aisles])))
        reorders = reorders[:-1]
        reorders = [list(map(int,reorder.split('_'))) for reorder in reorders]
        
        for aisle in all_aisles:
            label = -1 if mode == 'test' else aisle in next_aisles 
            dept = aisle_dept_dict[int(aisle)]
            user_aisle.append([user_id,int(aisle),dept])
            
            order_sizes_ord = []
            in_order_ord  =[]
            index_order_ord = []
            index_order_ratio_ord = []
            cumsum_inorder_ratio_ord = []
            cumsum_inorder_ord = []
            cnt_ord = []
            cumsum_cnt_ratio_ord = []
            cumsum_inorder,total_cnts = 0,0
            for idx,(order_aisle,reorder_aisle) in enumerate(zip(orders,reorders)):
                order_size = len(order_aisle)
                in_order = aisle in order_aisle
                order_set = list(unique_everseen(order_aisle))
                index_in_order = order_set.index(aisle) + 1 if in_order else 0
                index_in_order_ratio = index_in_order / len(order_set)
                total_orders = [1 if ele==aisle else 0 for ele in enumerate(order_aisle)]
                cumsum_inorder += in_order
                cumsum_inorder_ratio = cumsum_inorder / (idx + 1)
                cur_cnt = sum(total_orders)
                # total_cnts += cur_cnt
                # cumsum_cnt_ratio = total_cnts / (idx + 1)

                in_order_ord.append(in_order)
                index_order_ord.append(index_in_order)
                index_order_ratio_ord.append(index_in_order_ratio)
                cumsum_inorder_ord.append(cumsum_inorder)
                cumsum_inorder_ratio_ord.append(cumsum_inorder_ratio)
                order_sizes_ord.append(order_size)
                cnt_ord.append(cur_cnt)
                # cumsum_cnt_ratio_ord.append(cumsum_cnt_ratio)
            
            next_in_order = np.roll(in_order_ord,-1)
            next_in_order[-1] = label
            
            aisle_info = np.stack([in_order_ord,np.array(index_order_ord)/145,index_order_ratio_ord,
                                   cumsum_inorder_ratio_ord,order_days/30,
                                   np.array(order_sizes_ord)/145,cnt_ord,next_in_order],axis=1).astype(np.float16)
            aisle_info_dim = aisle_info.shape[1] - 1
            length = aisle_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,aisle_info_dim+1],dtype=np.float16)
            aisle_info = np.concatenate([aisle_info,paddings])
            
            data_dict[(user_id,int(aisle))] = (aisle_info,length)
            
    user_aisle = np.array(user_aisle)
    
    save_data  = [user_aisle,temporal_dict,data_dict]
    for path,file in zip(save_path,save_data):
        path = os.path.join(TMP_PATH,path)
        pickle_save_load(path,file,mode='save') 	
    
    return user_aisle,data_dict,temporal_dict,aisle_info_dim

def aisle_dataloader(inp,
                     data_dict,
                     temp_dict,
                     batch_size=32,
                     shuffle=True,
                     drop_last=False):
    def batch_gen():
        total_length = inp.shape[0]
        indices = np.arange(total_length)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0,total_length,batch_size):
            ind = indices[i:i+batch_size]
            if len(ind) < batch_size and drop_last:
                break
            else:
                batch = inp[ind]
                yield batch
    
    for batch in batch_gen():
        split_outputs = np.split(batch,batch.shape[1],axis=1)
        users,aisles,depts = list(map(np.squeeze,split_outputs))
        dows,hours,tzs = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,data_len = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(data_len)
        temporals = [dows,hours,tzs]
        yield full_batch,batch_lengths,temporals,users-1,aisles-1,depts-1

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()
class AisleTrainer(Trainer):
    def __init__(self,
                  data,
                  prod_data,
                  output_dim,
                  learning_rate=0.01,
                  lagging=1,
                  optim_option='adam',
                  batch_size=32,
                  warm_start=False,
                  early_stopping=3,
                  epochs=100,
                  eval_epoch=1):
        super().__init__(data=data,
                         prod_data=prod_data,
                         output_dim=output_dim,
                         learning_rate=learning_rate,
                         lagging=lagging,
                         optim_option=optim_option,
                         batch_size=batch_size,
                         warm_start=warm_start,
                         early_stopping=early_stopping,
                         epochs=epochs,
                         eval_epoch=eval_epoch)
        self.dataloader = aisle_dataloader
        self.data_maker = aisle_data_maker
        self.model = AisleLSTM
        
        self.emb_list = ['user_id','aisle_id','department_id']
        self.max_index_info = [data[col].max() for col in self.emb_list] + [self.max_len]
        self.aisle_dept_dict = self.prod_data.set_index('aisle_id')['department_id'].to_dict()
    
    def build_data_dl(self,mode):
        agg_data = self.build_agg_data('aisle_id')
        data_group = self.data_maker(agg_data,self.data,self.max_len,self.aisle_dept_dict,mode=mode)
        user_aisle,data_dict,temp_dict,aisle_dim = data_group
        for key,value in temp_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temp_dict[key] = value
        self.input_dim = self.temp_dim + aisle_dim + len(self.emb_list) * 50
        return [user_aisle,data_dict,temp_dict]
    
    def train(self,use_amp=False):
        core_info_tr= self.build_data_dl(mode='train')
        model,optimizer,lr_scheduler,start_epoch,best_loss,checkpoint_path = super().train(use_amp=use_amp)
        
        no_improvement = 0
        for epoch in range(start_epoch,self.epochs):
            total_loss,cur_iter = 0,1
            model.train()
            train_dl = self.dataloader(*core_info_tr,self.batch_size,True,True)
            train_batch_loader = tqdm(train_dl,total=core_info_tr[0].shape[0]//self.batch_size,desc=f'training next aisle basket at epoch:{epoch}',
                                      dynamic_ncols=True,leave=False)
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
                        _,preds = model(batch,*temps)
                        loss = self.loss_fn_tr(preds,label,batch_lengths)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _,preds = model(batch,*aux_info,*temps)
                    loss = self.loss_fn_tr(preds,label,batch_lengths)
                    loss.backward()
                    optimizer.step()

                cur_loss = loss.item()
                total_loss += cur_loss
                avg_loss = total_loss / cur_iter
                    
                cur_iter += 1
                train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
            
            if epoch % self.eval_epoch == 0:
                params = []
                eval_dl = self.dataloader(*core_info_tr,1024,False,False)
                eval_batch_loader = tqdm(eval_dl,total=core_info_tr[0].shape[0]//1024,desc='evaluating next aisle basket',
                                         leave=False,dynamic_ncols=True)
                model.eval()
                total_loss_val,ite = 0,1
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
                        
                        users,aisles,_  = aux_info
                        users,aisles = users.cpu().numpy(),aisles.cpu().numpy()
                        h = h.cpu().numpy()
                        h = h.astype(np.float16)
                        users += 1;aisles += 1
                        keys = np.stack([users,aisles],axis=1)
                        keys = list(map(tuple,(map(np.squeeze,np.split(keys,keys.shape[0],axis=0)))))
                        values = list(map(np.squeeze,np.split(h,h.shape[0],axis=0)))
                        param_list = list((map(lambda kv:(kv[0],kv[1]),zip(keys,values))))
                        params += param_list
                param_dict = dict(params)
                lr_scheduler.step(avg_loss_val)
                
                if avg_loss_val < best_loss:
                    best_loss = avg_loss_val
                    checkpoint = {'best_epoch':epoch,
                                  'best_loss':best_loss,
                                  'model_state_dict':model.state_dict(),
                                  'optimizer_state_dict':optimizer.state_dict()}
                    os.makedirs('checkpoint',exist_ok=True)
                    torch.save(checkpoint,checkpoint_path)
                    pickle_save_load('data/tmp/user_aisle_param.pkl',param_dict,mode='save') 
                    no_improvement = 0
                else:
                    no_improvement += 1
                del param_dict
                gc.collect()
                    
                if no_improvement == self.early_stopping:
                    logger.info('early stopping is trggered,the model has stopped improving')
                    return
    
if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
#     products = pd.read_csv('data/products.csv')

#     trainer = AisleTrainer(data,
#                            products,
#                            output_dim=20,
#                            lagging=1,
#                            learning_rate=0.002,
#                            optim_option='adam',
#                            batch_size=256,
#                            warm_start=False,
#                            early_stopping=2,
#                            epochs=100,
#                            eval_epoch=1)
#     trainer.train(use_amp=False)
    

#%%
import operator
from collections import defaultdict,Counter
class AisleStatsCollector(BaseStatsCollector):
    def __init__(self,
                 data):
        super().__init__(data=data)
        user_unique_aisles = data.groupby('user_id')['aisle_id'].apply(set).apply(list)
        self.unique_dict = user_unique_aisles.to_dict()
        self.stats = self.build_data()
    
    def build_data(self):
        stats = []
        
        order_aisle = self.data.groupby(['user_id','order_id','order_number','aisle_id'])['counter'].sum().reset_index()
        order_aisle = order_aisle.sort_values(['user_id','order_number'])
        order_aisle_ls = order_aisle.groupby(['user_id','order_id','order_number'])['aisle_id'].apply(list).reset_index()
        order_aisle_ls = order_aisle_ls.sort_values(['user_id','order_number'])
        del order_aisle_ls['order_number']
        stats.append(order_aisle_ls)
        
        order_aisle_shift = order_aisle_ls.groupby('user_id')['aisle_id'].shift(-1)
        order_aisle_shift.name = 'aisle_next'
        stats.append(order_aisle_shift)
        
        order_aisle_cnt = order_aisle.groupby(['user_id','order_id','order_number'])['counter'].apply(list).reset_index()
        order_aisle_cnt = order_aisle_cnt.sort_values(['user_id','order_number'])
        del order_aisle_cnt['order_number']
        stats.append(order_aisle_cnt[['counter']])
        
        for col in ['order_dow','days_since_prior_order']:
            s = self._shift_stats(col)
            stats.append(s)
        stats = pd.concat(stats,axis=1)
        stats = optimize_dtypes(stats)
        return stats
    
    def _shift_stats(self,col):
        stat = self.data.groupby(['user_id','order_id','order_number'])[col].apply(lambda x:x.iloc[-1]).reset_index()
        stat = stat.sort_values(['user_id','order_number'])
        del stat['order_number']
        stat = stat.groupby(['user_id'])[col].shift(-1)
        stat.name = col
        return stat
    
    def true_adjacent_stat(self,valid_cnt=1,min_interval=0,max_interval=31,comparator='equal'):
        operators = {'equal':operator.eq,
                     'greater_equal':operator.ge}
        op = operators[comparator]
        true_stats = defaultdict(lambda:defaultdict(int))
        true_interval_stats = defaultdict(list)
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc='building true adjacent row stat')
        for _,row in rows:
            user,cur_aisles,next_aisles,cnts = row['user_id'],row['aisle_id'],row['aisle_next'],row['counter']
            if not isinstance(next_aisles,list):
                continue
            order_interval = row['days_since_prior_order']
            unique_aisles = self.unique_dict[user]
            for aisle in unique_aisles:
                if aisle in cur_aisles :
                    index = cur_aisles.index(aisle)
                    cnt = cnts[index]
                    if op(cnt,valid_cnt) and order_interval >= min_interval and order_interval <  max_interval:
                        key = f'{str(cnt)}_{str(min_interval)}_{str(max_interval)}'
                        true_stats[aisle][f'total_{key}'] += 1
                        if aisle in next_aisles:
                            true_stats[aisle][f'in_{key}'] += 1
                            true_interval_stats[aisle].append(order_interval)
                            
        true_stats = self._convert_to_prob_dict(true_stats)
        for key,value in true_interval_stats.items():
            value = self._convert_list_to_dist(value)
            true_interval_stats[key] = value
        
        true_stats = self._check_nan(true_stats)
        return true_stats,true_interval_stats
    
aisle_collector = AisleStatsCollector(data=data)
aisle_data = aisle_collector.build_data()
        
#%%
# a,b = aisle_collector.true_adjacent_stat(valid_cnt=3,min_interval=7,max_interval=23,operator='equal')
a
#%%
from collections import Counter
import matplotlib.pyplot as plt
# user_unique_aisles = data.groupby('user_id')['aisle_id'].apply(set).apply(list)
# user_unique_aisles_dict = user_unique_aisles.to_dict()
# unique_aisles = data['aisle_id'].unique()

def _convert_to_prob_dict(x):
    df = pd.DataFrame.from_dict({k:dict(v) for k,v in x.items()},orient='index')
    col1,col2 = df.columns
    df = df[col2] / df[col1]
    df = df.to_dict()
    return df

def _convert_list_to_dist(x):
    counter = Counter(x)
    df = pd.DataFrame.from_dict(counter,orient='index')
    df /= df.sum()
    df = df.to_dict()
    return df[0]

def true_adjacent_stat(x,valid_cnt=1,min_interval=0,max_interval=31,comparator='equal'):
    operators = {'equal':operator.eq,
                 'greater_equal':operator.ge}
    op = operators[comparator]
    true_stats = defaultdict(lambda:defaultdict(int))
    true_interval_stats = defaultdict(list)
    rows = tqdm(x.iterrows(),total=x.shape[0],desc='building true adjacent row stat')
    for _,row in rows:
        user,cur_aisles,next_aisles,cnts = row['user_id'],row['aisle_id'],row['aisle_next'],row['counter']
        if not isinstance(next_aisles,list):
            continue
        order_interval = row['days_since_prior_order']
        unique_aisles = user_unique_aisles_dict[user]
        for aisle in unique_aisles:
            if aisle in cur_aisles :
                index = cur_aisles.index(aisle)
                cnt = cnts[index]
                if op(cnt,valid_cnt) and order_interval >= min_interval and order_interval <  max_interval:
                    key = f'{str(cnt)}_{str(min_interval)}_{str(max_interval)}'
                    true_stats[aisle][f'total_{key}'] += 1
                    if aisle in next_aisles:
                        true_stats[aisle][f'in_{key}'] += 1
                        true_interval_stats[aisle].append(order_interval)
                        
    true_stats = _convert_to_prob_dict(true_stats)
    for key,value in true_interval_stats.items():
        value = _convert_list_to_dist(value)
        true_interval_stats[key] = value
    return true_stats,true_interval_stats
    
a,b = true_adjacent_stat(aisle_data)
        
        
#%%

    
    