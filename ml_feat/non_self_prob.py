# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:38:41 2024

@author: congx
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict

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
#     # data = pd.read_csv('data/orders_info.csv')
    # df = data[data['reverse_order_number']>1]
    # unique_prod_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
#     # products = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
#     products = products.sort_values(['user_id','order_number'])
#     shift_1_products = products.groupby('user_id')['product_id'].shift(-1)
#     shift_1_products.name = 'next_products'
#     order_products = pd.concat([products,shift_1_products],axis=1)
#     order_products.dropna(axis=0,inplace=True)
    # aff_dict,aff_prob = affinity_product(order_products)
    
    # df = data[data['reverse_order_number']==1]
    # df = df.groupby('user_id')['product_id'].apply(list).reset_index()
    aff_stats = {}
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='collecting affinity stat')
    for _,row in rows:
        user = row['user_id']
        products = row['product_id']
        unq_prods = unique_prod_dict[user]
        for prod in unq_prods:
            probs = [aff_prob_dict.get((p,prod),0) if p!=prod else np.nan for p in products]
            minv = np.min(probs)
            maxv = np.max(probs)
            meanv = np.mean(probs)
            medianv = np.median(probs)
            stdv = np.std(probs)
            aff_stats[(user,prod)] = [minv,maxv,meanv,medianv,stdv]


#%%
# sorted_data = data.drop_duplicates(['user_id','order_id']).sort_values(['user_id','order_number'])
# order_days = sorted_data.groupby('user_id')['days_since_prior_order'].apply(list).reset_index()
# z = order_products.iloc[:10000]
# sorted_data.groupby(['user_id',''])
# x = pd.read_csv('metadata/probs_xxx_test.csv')
# z = x.iloc[:10000]
aff_prob_dict


# aff_prob_dict = {}
# for _,row in tqdm(x.iterrows(),total=x.shape[0],desc='building affinity prob dict'):
#     key = (row['product_id'],row['target_product'])
#     aff_prob_dict[key] = row['aff_prob']