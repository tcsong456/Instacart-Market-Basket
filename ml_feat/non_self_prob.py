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
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from utils.utils import optimize_dtypes

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
    
    aff_prob = pd.DataFrame.from_dict(aff_dict,orient='index',columns=['cnt']).reset_index()
    aff_prob['product_id'],aff_prob['target_product'] = zip(*aff_prob['index'])
    del aff_prob['index']
    aff_cnt = pd.DataFrame.from_dict(cnt_dict,orient='index',columns=['total']).reset_index().rename(columns={'index':'product_id'})
    aff_prob = aff_prob.merge(aff_cnt,how='left',on=['product_id'])
    aff_prob['aff_prob'] = aff_prob['cnt'] / aff_prob['total']
    aff_prob_dict = aff_prob.set_index(['product_id','target_product'])['aff_prob'].to_dict()
    return aff_prob_dict
                
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    unique_prod_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
    products = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    products = products.sort_values(['user_id','order_number'])
    shift_1_products = products.groupby('user_id')['product_id'].shift(-1)
    shift_1_products.name = 'next_products'
    order_products = pd.concat([products,shift_1_products],axis=1)
    order_products.dropna(axis=0,inplace=True)
    aff_prob_dict = affinity_product(order_products)
    
    df = data[data['reverse_order_number']==args.mode+1]
    df = df.groupby('user_id')['product_id'].apply(list).reset_index()
    aff_stats = {}
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='collecting affinity stat')
    for _,row in rows:
        user = row['user_id']
        products = row['product_id']
        unq_prods = unique_prod_dict[user]
        for prod in unq_prods:
            keys = [(p,prod) for p in products]
            probs = list(map(lambda key:aff_prob_dict.get(key,np.nan),keys))
            minv = np.nanmin(probs)
            maxv = np.nanmax(probs)
            meanv = np.nanmean(probs)
            medianv = np.nanmedian(probs)
            stdv = np.nanstd(probs)
            aff_stats[(user,prod)] = [minv,maxv,meanv,medianv,stdv]
    
    aff_stats = pd.DataFrame.from_dict(aff_stats,orient='index',columns=['min','max','mean','median','std']).add_prefix('aff_').reset_index()
    aff_stats['user_id'],aff_stats['product_id'] = zip(*aff_stats['index'])
    del aff_stats['index']
    suffix = 'test' if args.mode==0 else 'train'
    aff_stats = optimize_dtypes(aff_stats)
    aff_stats.to_csv(f'metadata/aff_probs_{suffix}.csv',index=False)

#%%