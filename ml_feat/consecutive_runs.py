# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:49:57 2024

@author: congx
"""
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def convertion(df):
    d = pd.DataFrame.from_dict(df)
    d = d[d['total']>=10]
    d.fillna(0,inplace=True)
    d['prob'] = d['cnt'] / d['total']
    ddict = d['prob'].to_dict()
    return ddict

def x_to_x(data,unq_dict):
    true_stat = defaultdict(lambda:defaultdict(int))
    fake_stat = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 1 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods = row['product_id'],row['shift_1_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                true_stat['total'][prod] += 1
                if prod in s1_prods:
                    true_stat['cnt'][prod] += 1
            else:
                fake_stat['total'][prod] += 1
                if prod in s1_prods:
                    fake_stat['cnt'][prod] += 1
    
    true_stat = convertion(true_stat)
    fake_stat = convertion(fake_stat)
    return true_stat,fake_stat

def xx_to_x(data,unq_dict):
    stat_111 = defaultdict(lambda:defaultdict(int))
    stat_101 = defaultdict(lambda:defaultdict(int))
    stat_011 = defaultdict(lambda:defaultdict(int))
    stat_001 = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 2 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods,s2_prods = row['product_id'],row['shift_1_products'],row['shift_2_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                if prod in s1_prods:
                    stat_111['total'][prod] += 1
                    if prod in s2_prods:
                        stat_111['cnt'][prod] += 1
                else:
                    stat_101['total'][prod] += 1
                    if prod in s2_prods:
                        stat_101['cnt'][prod] += 1
            else:
                if prod in s1_prods:
                    stat_011['total'][prod] += 1
                    if prod in s2_prods:
                        stat_011['cnt'][prod] += 1
                else:
                    stat_001['total'][prod] += 1
                    if prod in s2_prods:
                        stat_001['cnt'][prod] += 1
    
    stat_111 = convertion(stat_111)
    stat_101 = convertion(stat_101)
    stat_011 = convertion(stat_011)
    stat_001 = convertion(stat_001)
    return stat_111,stat_101,stat_011,stat_001 

def xxx_to_x(data,unq_dict):
    stat_1111 = defaultdict(lambda:defaultdict(int))
    stat_1101 = defaultdict(lambda:defaultdict(int))
    stat_1001 = defaultdict(lambda:defaultdict(int))
    stat_1011 = defaultdict(lambda:defaultdict(int))
    stat_0111 = defaultdict(lambda:defaultdict(int))
    stat_0101 = defaultdict(lambda:defaultdict(int))
    stat_0001 = defaultdict(lambda:defaultdict(int))
    stat_0011 = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 3 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods,s2_prods,s3_prods = row['product_id'],row['shift_1_products'],row['shift_2_products'],row['shift_3_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                if prod in s1_prods:
                    if prod in s2_prods:
                        stat_1111['total'][prod] += 1
                        if prod in s3_prods:
                            stat_1111['cnt'][prod] += 1
                    else:
                        stat_1101['total'][prod] += 1
                        if prod in s3_prods:
                            stat_1101['cnt'][prod] += 1
                else:
                    if prod in s2_prods:
                        stat_1011['total'][prod] +=1
                        if prod in s3_prods:
                            stat_1011['cnt'][prod] += 1
                    else:
                        stat_1001['total'][prod] += 1
                        if prod in s3_prods:
                            stat_1001['cnt'][prod] += 1
            else:
                if prod in s1_prods:
                    if prod in s2_prods:
                        stat_0111['total'][prod] += 1
                        if prod in s3_prods:
                            stat_0111['cnt'][prod] += 1
                    else:
                        stat_0101['total'][prod] += 1
                        if prod in s3_prods:
                            stat_0101['cnt'][prod] += 1
                else:
                    if prod in s2_prods:
                        stat_0011['total'][prod] += 1
                        if prod in s3_prods:
                            stat_0011['cnt'][prod] += 1
                    else:
                        stat_0001['total'][prod] += 1
                        if prod in s3_prods:
                            stat_0001['cnt'][prod] += 1
                            
    stat_1111 = convertion(stat_1111)
    stat_1101 = convertion(stat_1101)
    stat_1001 = convertion(stat_1001)
    stat_1011 = convertion(stat_1011)
    stat_0111 = convertion(stat_0111)
    stat_0101 = convertion(stat_0101)
    stat_0001 = convertion(stat_0001)
    stat_0011 = convertion(stat_0011)
    return stat_1111,stat_1101,stat_1001,stat_1011,stat_0111,stat_0101,stat_0001,stat_0011
                            
      
if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    data = data[data['reverse_order_number']>0]
    unique_data_dict = data.groupby('user_id')['product_id'].apply(set).to_dict()
    
    user_prods = data.groupby(['user_id','order_id'])['product_id'].apply(list).reset_index()
    prods_shift_1 = user_prods.groupby('user_id')['product_id'].shift(-1)
    prods_shift_1.name = 'shift_1_products'
    adjacent_basket_1 = pd.concat([user_prods,prods_shift_1],axis=1)
    adjacent_basket_1.dropna(axis=0,inplace=True)
    stat_11,stat_01 = x_to_x(adjacent_basket_1,unique_data_dict)
    
    prods_shift_2 = user_prods.groupby('user_id')['product_id'].shift(-2)
    prods_shift_2.name = 'shift_2_products'
    adjacent_basket_2 = pd.concat([user_prods,prods_shift_1,prods_shift_2],axis=1)
    adjacent_basket_2.dropna(axis=0,inplace=True)
    stat_111,stat_101,stat_011,stat_001 = xx_to_x(adjacent_basket_2,unique_data_dict)
    
    prods_shift_3 = user_prods.groupby('user_id')['product_id'].shift(-3)
    prods_shift_3.name = 'shift_3_products'
    adjacent_basket_3 = pd.concat([user_prods,prods_shift_1,prods_shift_2,prods_shift_3],axis=1)
    adjacent_basket_3.dropna(axis=0,inplace=True)
    stat_1111,stat_1101,stat_1001,stat_1011,stat_0111,stat_0101,stat_0001,stat_0011 = xxx_to_x(adjacent_basket_3,unique_data_dict)

#%%
# from utils.utils import optimize_dtypes
# single_true_stat,single_fake_stat = x_to_x(adjacent_basket_1,unique_data_dict)
# x = pd.read_csv('metadata/prod_interval_stats_test.csv')
# adjacent_basket_3
# optimize_dtypes(x).dtypes

