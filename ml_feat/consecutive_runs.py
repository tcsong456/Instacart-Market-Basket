# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:49:57 2024

@author: congx
"""
import gc
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    unique_data_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
    
    user_prods = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    user_prods = user_prods.sort_values(['user_id','order_number'])
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
    
    probs = dict()
    df = data[data['reverse_order_number']==args.mode+1]
    target_1_grp = df.groupby('user_id')['product_id'].apply(list).reset_index()
    rows = tqdm(target_1_grp.iterrows(),total=target_1_grp.shape[0],desc='collecting stats for two consecutive run of products')
    for _,row in rows:
        user,products = row['user_id'],row['product_id']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in products:
                prob = stat_11.get(prod,np.nan)
            else:
                prob = stat_01.get(prod,np.nan)
            probs[(user,prod)] = prob

    probs_xx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xx']).reset_index()
    probs_xx['user_id'],probs_xx['product_id'] = zip(*probs_xx['index'])
    del probs_xx['index']

    probs = dict()
    df = data[data['reverse_order_number']==args.mode+2]
    target_2_grp = df.groupby('user_id')['product_id'].apply(list).reset_index().rename(columns={'product_id':'shift_1_products'})
    target_grp = pd.merge(target_2_grp,target_1_grp,how='left',on=['user_id'])
    rows = tqdm(target_grp.iterrows(),total=target_grp.shape[0],desc='collecting stats for three consecutive run of products')
    for _,row in rows:
        user,products,s1_products = row['user_id'],row['product_id'],row['shift_1_products']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in s1_products:
                if prod in products:
                    prob = stat_111.get(prod,np.nan)
                else:
                    prob = stat_101.get(prod,np.nan)
            else:
                if prod in products:
                    prob = stat_011.get(prod,np.nan)
                else:
                    prob = stat_001.get(prod,np.nan)
            probs[(user,prod)] = prob
    probs_xxx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xxx']).reset_index()
    probs_xxx['user_id'],probs_xxx['product_id'] = zip(*probs_xxx['index'])
    del probs_xxx['index']
    
    probs = dict()
    df = data[data['reverse_order_number']==args.mode+3]
    target_3_grp = df.groupby('user_id')['product_id'].apply(list).reset_index().rename(columns={'product_id':'shift_2_products'})
    target_grp = target_3_grp.merge(target_2_grp,how='left',on=['user_id']).merge(target_1_grp,how='left',on=['user_id'])
    target_grp.dropna(axis=0,inplace=True)
    rows = tqdm(target_grp.iterrows(),total=target_grp.shape[0],desc='collecting stats for three consecutive run of products')
    for _,row in rows:
        user,products,s1_products,s2_products = row['user_id'],row['product_id'],row['shift_1_products'],row['shift_2_products']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in s2_products:
                if prod in s1_products:
                    if prod in products:
                        prob = stat_1111.get(prod,np.nan)
                    else:
                        prob = stat_1101.get(prod,np.nan)
                else:
                    if prod in s1_products:
                        prob = stat_1011.get(prod,np.nan)
                    else:
                        prob = stat_1001.get(prod,np.nan)
            else:
                if prod in s1_products:
                    if prod in s2_products:
                        prob = stat_0111.get(prod,np.nan)
                    else:
                        prob = stat_0101.get(prod,np.nan)
                else:
                    if prod in s2_products:
                        prob = stat_0011.get(prod,np.nan)
                    else:
                        prob = stat_0001.get(prod,np.nan)
            probs[(user,prod)] = prob
    probs_xxxx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xxxx']).reset_index()
    probs_xxxx['user_id'],probs_xxxx['product_id'] = zip(*probs_xxxx['index'])
    del probs_xxxx['index']
    
    suffix = 'test' if args.mode==0 else 'train'
    probs = probs_xx.merge(probs_xxx,how='left',on=['user_id','product_id']).merge(probs_xxxx,how='left',on=['user_id','product_id'])
    probs.to_csv(f'metadata/probs_xxx_{suffix}.csv',index=False)

#%%



