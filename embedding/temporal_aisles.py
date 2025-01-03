# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:50:57 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from utils.utils import pad,pickle_save_load
from nn_model.aisle_lstm import AisleLSTM
from embedding.trainer import Trainer
from more_itertools import unique_everseen

def aisle_data_maker(agg_data,max_len,aisle_dept_dict,mode='train'):
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
        temporal_dict[user_id] = [order_dow,order_hour,order_tz,pad(order_days,max_len)]
        
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
            aisle_var_ratio_ord = []
            reorder_ratio_ord = []
            reorder_basket_ratio_ord = []
            reorder_total_ratio_ord = []
            cumsum_inorder,total_cnts,total_sizes,total_reorder_sums = 0,0,0,0
            seen_aisles = set()
            for idx,(order_aisle,reorder_aisle) in enumerate(zip(orders,reorders)):
                order_size = len(order_aisle)
                in_order = aisle in order_aisle
                order_set = list(unique_everseen(order_aisle))
                index_in_order = order_set.index(aisle) + 1 if in_order else 0
                index_in_order_ratio = index_in_order / len(order_set)
                total_orders = [1 if ele==aisle else 0 for ele in order_aisle]
                cumsum_inorder += in_order
                cumsum_inorder_ratio = cumsum_inorder / (idx + 1)
                cur_cnt = sum(total_orders)
                total_cnts += cur_cnt
                cumsum_cnt_ratio = total_cnts / (idx + 1)
                reorder_total = sum(reorder_aisle)
                total_reorder_sums += reorder_total
                reorder_ratio = total_reorder_sums / (idx + 1)
                reorder_basket_ratio = reorder_total / order_size
                aisle_var_ratio_ord.append(len(seen_aisles)/(idx+1))
                seen_aisles |= set(order_aisle)
                total_sizes += order_size
                reorder_total_ratio = total_reorder_sums / total_sizes

                in_order_ord.append(in_order)
                index_order_ord.append(index_in_order)
                index_order_ratio_ord.append(index_in_order_ratio)
                cumsum_inorder_ord.append(cumsum_inorder)
                cumsum_inorder_ratio_ord.append(cumsum_inorder_ratio)
                order_sizes_ord.append(order_size)
                cnt_ord.append(cur_cnt)
                cumsum_cnt_ratio_ord.append(cumsum_cnt_ratio)
                reorder_ratio_ord.append(reorder_ratio)
                reorder_basket_ratio_ord.append(reorder_basket_ratio)
                reorder_total_ratio_ord.append(reorder_total_ratio)
            
            next_in_order = np.roll(in_order_ord,-1)
            next_in_order[-1] = label
            
            aisle_info = np.stack([in_order_ord,np.array(index_order_ord)/145,index_order_ratio_ord,aisle_var_ratio_ord,
                                   reorder_ratio_ord,cumsum_inorder_ratio_ord,order_days/30,cumsum_cnt_ratio_ord,
                                   reorder_basket_ratio_ord,np.array(order_sizes_ord)/145,cnt_ord,reorder_total_ratio_ord,
                                   next_in_order],axis=1).astype(np.float16)
            aisle_info_dim = aisle_info.shape[1] - 1
            length = aisle_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,aisle_info_dim+1],dtype=np.float16)
            aisle_info = np.concatenate([aisle_info,paddings])
            
            data_dict[(user_id,int(aisle))] = (aisle_info,length)
            
    user_aisle = np.array(user_aisle)
    pickle_save_load(f'data/tmp/temporal_dict_{mode}.pkl',temporal_dict,mode='save') 	
    
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
        dows,hours,tzs,days = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,data_len = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(data_len)
        temporals = [dows,hours,tzs,days]
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
        self.model_name = self.model.__name__
        self.attr = 'aisle'
        
        self.emb_list = ['user_id','aisle_id','department_id']
        self.max_index_info = [data[col].max() for col in self.emb_list] + [self.max_len]
        aisle_dept_dict = self.prod_data.set_index('aisle_id')['department_id'].to_dict()
        self.data_maker_dicts = [aisle_dept_dict]
    
    def build_data_dl(self,mode):
        agg_data = self.build_agg_data('aisle_id')
        data_group = self.data_maker(agg_data,self.max_len,*self.data_maker_dicts,mode=mode)
        user_aisle,data_dict,temp_dict,aisle_dim = data_group
        for key,value in temp_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temp_dict[key] = value
        self.input_dim = self.temp_dim + aisle_dim + len(self.emb_list) * 50
        self.agg_data = agg_data
        return [user_aisle,data_dict,temp_dict]
    
if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')

    trainer = AisleTrainer(data,
                            products,
                            output_dim=50,
                            lagging=1,
                            learning_rate=0.002,
                            optim_option='adam',
                            batch_size=256,
                            warm_start=False,
                            early_stopping=2,
                            epochs=10,
                            eval_epoch=1)
    trainer.train(use_amp=False)
    trainer.predict(save_name='user_aisle_pred')

#%%
