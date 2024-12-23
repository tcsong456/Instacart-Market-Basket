# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:24:54 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import argparse
import warnings
warnings.filterwarnings(action='ignore')
import bisect
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.utils import collect_stats,optimize_dtypes

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

def interval_days_collector(df,tdf,interval_stats):
    user_prods = df.groupby('user_id')['product_id'].apply(set).reset_index()
    user_prods = user_prods.explode('product_id')
    tdf = tdf.drop_duplicates(['user_id','days_since_prior_order'])[['user_id','days_since_prior_order']]
    user_prods = pd.merge(user_prods,tdf,how='left',on=['user_id'])
    user_prods = user_prods.merge(interval_stats,how='left',on=['product_id'])
    user_prods['interval_mean_diff'] = user_prods['days_since_prior_order'] - user_prods['prod_mean']
    user_prods['interval_median_diff'] = user_prods['days_since_prior_order'] - user_prods['prod_median']
    user_prods['mean_diff_ratio'] = user_prods['interval_mean_diff'] / user_prods['prod_mean']
    user_prods['median_diff_ratio'] = user_prods['interval_median_diff'] / user_prods['prod_median'] 
    user_prods.loc[np.isinf(user_prods['mean_diff_ratio']),'mean_diff_ratio'] = np.nan
    user_prods.loc[np.isinf(user_prods['median_diff_ratio']),'median_diff_ratio'] = np.nan
    user_prods = optimize_dtypes(user_prods)
    del user_prods['days_since_prior_order']
    
    return user_prods
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    target_data = data[data['reverse_order_number']==args.mode]
    data = data[data['reverse_order_number']>args.mode]
    data = data.sort_values(['user_id','order_number'])
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
    suffix = 'test' if args.mode==0 else 'train'
    user_prod_interval_days = interval_days_collector(data,target_data,product_interval_stats)
    user_prod_interval_days.to_csv(f'metadata/prod_interval_stats_{suffix}.csv',index=False)

#%%


