# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:50:57 2024

@author: congx
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain

def make_data(data,mode='train'):
    dpar = tqdm(data.iterrows(),total=len(data),desc='building aisle data for dataloader',
                dynamic_ncols=True,leave=False)
    for _,row in dpar:
        user_id,aisles = row['user_id'],row['aisle_id']
        temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
        if mode == 'train' and row['eval_set'] == 'test':
            aisles = aisles[:-1]
            for idx,col_data in enumerate(temporal_cols):
                col_data = col_data[:-1]
                temporal_cols[idx] = col_data
            order_dow,order_hour,order_tz,order_days = temporal_cols
        
        aisles,next_aisles = aisles[:-1],aisles[-1]
        orders = [aisle.split('_') for aisle in aisles]
        all_aisles = list(set(chain.from_iterable([aisle.split('_') for aisle in aisles])))
        
        
        for aisle in all_aisles:
            label = aisle in next_aisles
            
            order_sizes_ord = []
            in_order_ord  =[]
            index_order_ord = []
            index_order_ratio = []
            avg_index_ord = []
            cnt_ratio_ord = []
            avg_index_ord = []
            tendency_ord = []
            total_cnts,total_in_order,total_size = 0,0,0
            for idx,order_aisle in enumerate(orders):
                order_aisles = order_aisle.split('_')
                order_size = len(order_aisles)
                in_order = aisle in order_aisles
                index_in_order = order_aisles.index(aisle) if in_order else 0
                index_in_order_ratio = index_in_order / order_size
                aisle_cnts = (np.array(order_aisles)==aisle).sum() if in_order else 0
                reorder_size_ratio = aisle_cnts / order_size
                total_cnts += aisle_cnts
                cnt_ratio_order = total_cnts / (idx + 1)
                avg_pos = [idx+1 for idx,ele in enumerate(order_aisles) if ele==aisle] if in_order else 0
                avg_pos_ratio = avg_pos / order_size
                total_in_order += in_order
                in_order_ratio = total_in_order / (idx + 1)
                total_size += order_size
                reorder_tendency = total_cnts / total_size
                
                
                


#%%
# import torch
# data = pd.read_pickle('data/tmp/user_product_info.csv')
# z = data.iloc[:1000]
# # checkpoint = torch.load('checkpoint/ProdLSTM_best_checkpoint.pth')
# # orders = pd.read_csv('data/orders_info.csv')
# # c = orders.iloc[:10000]
# # aisles = z.loc[0,'aisle_id']
# # y = aisles[0].split('_')
# sum([True,Trued\
# data[data['product_id'].map(lambda x:len(x)-3<2)]
# data_tr = 
# data_tr[data_tr['product_id'].map(lambda x:len(x)-1>=2)]