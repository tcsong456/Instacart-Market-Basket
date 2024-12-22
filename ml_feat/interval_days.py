# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:24:54 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import warnings
warnings.filterwarnings(action='ignore')
import bisect
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.utils import collect_stats

def stat_interval_days(df):
    prod_intervals = defaultdict(list)
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building interval days dictionary')
    for _,row in rows:
        product,ord_num_max = row['product_id'],row['order_number_max']
        interval_days = np.array(row['days_since_prior_order'])
        order_numbers = row['order_number']
        if 1 not in interval_days:
            bisect.insort(order_numbers,1)
        if ord_num_max not in interval_days:
            bisect.insort(order_numbers,ord_num_max)
        start_order_numbers,end_order_numbers = order_numbers[:-1],order_numbers[1:]
        intervals = []
        
        for start,end in zip(start_order_numbers,end_order_numbers):
            indices = np.arange(start,end)
            intl = sum(interval_days[indices])
            intervals.append(intl)
        prod_intervals[product] += intervals
    
    dict_items = tqdm(prod_intervals.items(),total=len(prod_intervals),desc='collecting stats of intervals')
    for key,value in dict_items:
        stats = collect_stats(value,['min','max','mean','median','std'])
        prod_intervals[key] = stats
    return prod_intervals

if __name__ == '__main__':
    data = pd.read_csv('data/data_info.csv')
    data = data[data['reverse_order_nubmer']>0]
    order_numbers = data.groupby(['user_id','product_id'])['order_number'].apply(list).reset_index()
    sorted_data = data.drop_duplicates(['user_id','order_id']).sort_values(['user_id','order_number'])
    order_days = sorted_data.groupby('user_id')['days_since_prior_order'].apply(list).reset_index()
    order_info = pd.merge(order_numbers,order_days,how='left',on='user_id')
    max_order_numbers = data.groupby('user_id')['order_number'].max().reset_index().rename(columns={'order_number':'order_number_max'})
    order_info = order_info.merge(max_order_numbers,how='left',on=['user_id'])
    del sorted_data,order_days,max_order_numbers
    gc.collect()
    prod_intervals = stat_interval_days(order_info)
    product_interval_stats = pd.DataFrame.from_dict(prod_intervals,orient='index',columns=['min','max','mean','median','std']
                                                    ).add_prefix('prod_').reset_index().rename(columns={'index':'product_id'})
    product_interval_stats.to_csv('data/tmp/product_interval_stats.csv')
    

#%%
