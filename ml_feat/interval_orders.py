# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:06:23 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.utils import collect_stats

def interval_orders(df):
    intervals = defaultdict(list)
    df_intervals = defaultdict(list)
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building interval orders')
    for _,row in rows:
        orn = np.array(row['order_number'])
        product = row['product_id']
        if len(orn) == 1:
            continue
        prod_start,prod_end = orn[:-1],orn[1:]
        interval = prod_end - prod_start
        for intl in interval:
            intervals[product].append(intl)
        
    intl_dict = tqdm(intervals.items(),total=len(intervals.keys()),desc='collecting interval stats')
    for key,value in intl_dict:
        stats = collect_stats(value,['min','max','mean','median','std'])
        df_intervals[key] = stats
    df_intervals = pd.DataFrame.from_dict(df_intervals,orient='index',columns=['min','max','mean','median','std']).add_prefix('interval_')
    return df_intervals
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    user_prod_orn = df.groupby(['user_id','product_id'])['order_number'].apply(list).reset_index()
    intervals = interval_orders(user_prod_orn).reset_index().rename(columns={'index':'product_id'})
    
    df = data[(data['reverse_order_number']>args.mode)&(data['reverse_order_number']<=args.mode+5)]
    user_prod_orn = df.groupby(['user_id','product_id'])['order_number'].apply(list).reset_index()
    intervals_5 = interval_orders(user_prod_orn).add_suffix('_5').reset_index().rename(columns={'index':'product_id'})
    intervals = pd.merge(intervals,intervals_5,how='left',on=['product_id'])
    
    suffix = 'test' if args.mode==0 else 'train'
    intervals.to_csv(f'metadata/intervals_{suffix}.csv',index=False)


#%%

