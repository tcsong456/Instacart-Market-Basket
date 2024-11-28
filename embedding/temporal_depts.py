# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:03:51 2024

@author: congx
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.utils import pad,pickle_save_load

def depts_data_maker(data,max_len,mode='train'):
    suffix = mode + '.pkl'
    temp_dict = pickle_save_load('data/tmp/temporal_dict_{suffix}',mode='load')
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building department data for dataloader')
    for _,row in rows:
        reorders = row['reordered']
        user,departments,eval_set = row['user_id'],row['department_id'],row['eval_set']
        temporals = [row['order_hour_of_day'],row['order_dow'],row['time_zone'],row['days_since_prior_order']]
        if eval_set == 'test' and mode != 'test':
            departments = departments[:-1]
            reorders = reorders[:-1]
        
        departments,next_departments = departments[:-1],departments[-1]
        next_departments = next_departments.split('_')
        orders = [dept.split('_') for dept in departments]
        reorders = reorders[:-1]
        reorders = [list(map(int(reorder.split('_')))) for reorder in reorders]
        
        
        
            
        

#%%
agg_data = pd.read_pickle('data/tmp/user_product_info.csv')
z = agg_data.iloc[:1000]