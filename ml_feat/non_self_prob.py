# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:38:41 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from functools import partial
from utils.utils import collect_stats

def affinity_product(df):
    aff_dict = defaultdict(int)
    cnt_dict = defaultdict(int)
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building affinity data')
    for _,row in rows:
        prods,next_prods = row['product_id'],row['next_products']
        for prod in next_prods:
            p = prods.copy()
            if prod in p:
                p.remove(prod)
            combs = product(p,[prod])
            for comb in combs:
                aff_dict[comb] += 1
        for pdd in prods:
            cnt_dict[pdd] += 1
    
    aff_prob = pd.DataFrame.from_dict(aff_dict,orient='index',columns=['aff_cnt']).reset_index()
    aff_prob['product_id'],aff_prob['target_product'] = zip(*aff_prob['index'])
    del aff_prob['index']
    aff_cnt = pd.DataFrame.from_dict(cnt_dict,orient='index',columns=['total']).reset_index().rename(columns={'index':'product_id'})
    aff_prob = aff_prob.merge(aff_cnt,how='left',on=['product_id'])
    aff_prob['aff_prob'] = aff_prob['aff_cnt'] / aff_prob['total']
    aff_all_dict = aff_prob.set_index(['product_id','target_product'])[['aff_prob','aff_cnt']].apply(list,axis=1).to_dict()
    # aff_prob_dict = aff_prob.set_index(['product_id','target_product'])['aff_prob'].to_dict()
    # aff_cnt_dict = aff_prob.set_index(['product_id','target_product'])['aff_cnt'].to_dict()
    return aff_all_dict

def affinity_collector(df,aff_dict,unique_prod_dict):
    aff_stats = {}
    agg_func = ['nanmin','nanmax','nanmean','nanmedian','nanstd']
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='collecting affinity stat')
    for _,row in rows:
        user = row['user_id']
        products = row['product_id']
        unq_prods = unique_prod_dict[user]
        for prod in unq_prods:
            keys = [(p,prod) for p in products]
            # probs = list(map(lambda key:aff_prob_dict.get(key,np.nan),keys))
            # cnts = list(map(lambda key:aff_cnt_dict.get(key,np.nan),keys))
            probs,cnts = zip(*list(map(lambda key:aff_dict.get(key,[np.nan,np.nan]),keys)))
            prob_stats = collect_stats(probs,agg_func=agg_func)
            cnt_stats = collect_stats(cnts,agg_func=agg_func)
            stats = prob_stats + cnt_stats
            aff_stats[(user,prod)] = stats
    
    stat_funcs = ['min','max','mean','median','std']
    prob_cols = cols(stat_funcs,'aff_prob')
    cnt_cols = cols(stat_funcs,'aff_cnt')
    aff_stats = pd.DataFrame.from_dict(aff_stats,orient='index',columns=prob_cols+cnt_cols).reset_index()
    aff_stats['user_id'],aff_stats['product_id'] = zip(*aff_stats['index'])
    del aff_stats['index']
    return aff_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    # parser.add_argument('--cores',default=4,type=int)
    args = parser.parse_args()
    
    cols = lambda x,prefix:list(map(lambda k:prefix+'_'+k,x))
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    unique_prod_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
    products = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    products = products.sort_values(['user_id','order_number'])
    shift_1_products = products.groupby('user_id')['product_id'].shift(-1)
    shift_1_products.name = 'next_products'
    order_products = pd.concat([products,shift_1_products],axis=1)
    order_products.dropna(axis=0,inplace=True)
    aff_dict = affinity_product(order_products)
    
    df = data[data['reverse_order_number']==args.mode+1]
    df = df.groupby('user_id')['product_id'].apply(list).reset_index()
    aff_stats = affinity_collector(df,aff_dict,unique_prod_dict)
    # rows = df.shape[0]
    # batch = rows // args.cores
    # batches = []
    # for i in range(args.cores-1):
    #     start = i * batch
    #     end = start + batch
    #     batches.append(df.iloc[start:end])
    # batches.append(df.iloc[end:])
    # func = partial(affinity_collector,aff_dict=aff_dict,unique_prod_dict=unique_prod_dict)
    # with Pool(args.cores) as pool:
    #     results = pool.map(func,batches)
    # aff_stats = pd.concat(results)

    suffix = 'test' if args.mode==0 else 'train'
    aff_stats.to_csv(f'metadata/aff_{suffix}.csv',index=False)

#%%

