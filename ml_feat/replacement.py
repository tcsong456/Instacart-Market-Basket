# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:29:16 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from utils.utils import collect_stats

def stat_replacement(df):
    r = defaultdict(lambda:defaultdict(int))
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building stats for replacement')
    for _,row in rows:
        cur_prods,s1_prods,s2_prods = row['product_id'],row['shift_1_product'],row['shift_2_product']
        replacements = list(set(s1_prods) - (set(cur_prods) | set(s2_prods)))
        sources = list(set(cur_prods) - set(s1_prods))
        buy_backs = list(set(cur_prods) & set(s2_prods) - set(s1_prods))
        if len(replacements) > 0 and len(sources) > 0:
            cnt = product(sources,replacements)
            for key in cnt:
                r['total'][key] += 1
            bb = product(buy_backs,replacements)
            for key in bb:
                r['cnt'][key] += 1
    return r

def replacement_collector(df,stat_dict,mode='train'):
    rpl_stats = defaultdict(list)
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='collecting stats for replacements')
    for _,row in rows:
         user= row['user_id']
         cur_prods,prods_shf1,prods_shf2 = row['product_id'],row['products_shift_1'] ,row['products_shift_2'] 
         src_prods = set(cur_prods) - set(prods_shf1)
         dst_prods = set(prods_shf1) - (set(cur_prods)|set(prods_shf2))
         if len(src_prods) > 0 and len(dst_prods) > 0:
             for src in src_prods:
                 stats = []
                 keys = product([src],dst_prods)
                 for key in keys:
                     s = stat_dict.get(key,np.nan)
                     stats.append(s)
                 if np.isnan(sum(stats)):
                     stats = [np.nan] * 5
                 else:
                     stats = collect_stats(stats,['nanmin','nanmax','nanmean','nanmedian','nanstd'])
                 rpl_stats[(user,src)] = stats
    return rpl_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    cur_prods = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    cur_prods = cur_prods.sort_values(['user_id','order_number'])
    shift_1_prods = cur_prods.groupby('user_id')['product_id'].shift(-1)
    shift_1_prods.name = 'shift_1_product'
    shift_2_prods = cur_prods.groupby('user_id')['product_id'].shift(-2)
    shift_2_prods.name = 'shift_2_product'
    order_products = pd.concat([cur_prods,shift_1_prods,shift_2_prods],axis=1)
    order_products.dropna(axis=0,inplace=True)
    del cur_prods,shift_1_prods,shift_2_prods
    gc.collect()
    
    buy_backs = stat_replacement(order_products)
    buy_backs = pd.DataFrame.from_dict(buy_backs).reset_index().fillna(0)
    buy_backs = buy_backs[buy_backs['total']>=10]
    buy_backs['bb_ratio'] = buy_backs['cnt'] / buy_backs['total']
    buy_backs = buy_backs.rename(columns={'level_0':'src_prod','level_1':'rpl_prod'}).astype(np.float32)
    buy_backs_ratio = buy_backs.set_index(['src_prod','rpl_prod'])['bb_ratio'].to_dict()
    buy_backs = buy_backs.set_index(['src_prod','rpl_prod'])['cnt'].to_dict()
    
    ord_num_end = args.mode + 2
    rpl_data = data[(data['reverse_order_number']>=args.mode)&(data['reverse_order_number']<=ord_num_end)] 
    rpl_data = rpl_data.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    rpl_data = rpl_data.sort_values(['user_id','order_number'])
    rpl_prods_shf1 = rpl_data.groupby('user_id')['product_id'].shift(-1)
    rpl_prods_shf1.name = 'products_shift_1'
    rpl_prods_shf2 = rpl_data.groupby('user_id')['product_id'].shift(-2)
    rpl_prods_shf2.name = 'products_shift_2'
    rpl_prods = pd.concat([rpl_data,rpl_prods_shf1,rpl_prods_shf2],axis=1)
    rpl_prods.dropna(axis=0,inplace=True)
    del rpl_prods_shf1,rpl_prods_shf2,rpl_data
    gc.collect()
    
    rpl_dict = replacement_collector(rpl_prods,buy_backs)
    rpl_stats = pd.DataFrame.from_dict(rpl_dict,orient='index',columns=['rpl_min','rpl_max','rpl_mean','rpl_median','rpl_std']).reset_index()
    rpl_stats['user_id'],rpl_stats['product_id'] = zip(*rpl_stats['index'])
    del rpl_stats['index']
    
    rpl_dict_ratio = replacement_collector(rpl_prods,buy_backs_ratio)
    rpl_stats_ratio = pd.DataFrame.from_dict(rpl_dict_ratio,orient='index',columns=['rpl_min','rpl_max','rpl_mean','rpl_median','rpl_std'])
    rpl_stats_ratio = rpl_stats_ratio.add_suffix('_ratio').reset_index()
    rpl_stats_ratio['user_id'],rpl_stats_ratio['product_id'] = zip(*rpl_stats_ratio['index'])
    del rpl_stats_ratio['index']
    
    rpl_stats = pd.merge(rpl_stats,rpl_stats_ratio,on=['user_id','product_id'])
    
    suffix = 'test' if args.mode==0 else 'train'
    path = f'metadata/rpl_prods_{suffix}.csv'
    rpl_stats.to_csv(path,index=False)


#%%



