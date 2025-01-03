# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:45:13 2024

@author: congx
"""
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def convert(df,col,prob_col):
    temp_prob_dict = defaultdict(lambda:defaultdict(float))
    temp_cnt_dict = defaultdict(lambda:defaultdict(float))
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='converting df to dictionary')
    for _,row in rows:
        prod,tmp_col,prob = row['product_id'],row[col],row[prob_col] 
        temp_prob_dict[prod][tmp_col] = prob
        temp_cnt_dict[prod][tmp_col] = row['cnt']
    return temp_prob_dict,temp_cnt_dict

def build_tmp(df,col):
    attr_sum = df.groupby(['product_id',col])['counter'].sum().reset_index().rename(columns={'counter':'cnt'})
    user_sum = df.groupby(['product_id'])['counter'].sum().reset_index().rename(columns={'counter':'total'})
    attr = attr_sum.merge(user_sum,how='left',on=['product_id'])
    attr[f'{col}_prob'] = attr['cnt'] / attr['total']
    return attr

def build_up_tmp(df,col):
    user_prod_attr = df.groupby(['user_id','product_id',col])['counter'].sum().reset_index().rename(columns={'counter':'cnt'})
    prod_attr = df.groupby(['user_id','product_id'])['counter'].sum().reset_index().rename(columns={'counter':'total'})
    user_prod_attr = user_prod_attr.merge(prod_attr,how='left',on=['user_id','product_id'])
    user_prod_attr[f'{col}_dist'] = user_prod_attr['cnt'] / user_prod_attr['total']
    return user_prod_attr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    data = pd.read_csv('data/orders_info.csv')
    suffix = 'test' if args.mode==0 else 'train'
    data_tr = data[data['cate']=='train']
    data_te = data[data['cate']=='test']
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']>0]
        df_te = data_te[data_te['reverse_order_number']>1]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']>0]

    df['counter'] = 1
    unique_prod_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
    dow_data = build_tmp(df,'order_dow')
    dow_prob_dict,dow_cnt_dict = convert(dow_data,'order_dow','order_dow_prob')
    
    tz_data = build_tmp(df,'time_zone')
    tz_prob_dict,tz_cnt_dict = convert(tz_data,'time_zone','time_zone_prob')
    
    temporal_dict = {}
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']==0]
        df_te = data_te[data_te['reverse_order_number']==1]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']==0]
    df = df.drop_duplicates(['user_id','order_id'])
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='collecting temporal data')
    for _,row in rows:
        user = row['user_id']
        dow,tz = row['order_dow'],row['time_zone']
        for prod in unique_prod_dict[user]:
            dow_prob = dow_prob_dict[prod].get(dow,np.nan)
            tz_prob = tz_prob_dict[prod].get(tz,np.nan)
            dow_cnt = dow_cnt_dict[prod].get(dow,np.nan)
            tz_cnt = tz_cnt_dict[prod].get(tz,np.nan)
            temporal_dict[(user,prod)] = [dow_prob,dow_cnt,tz_prob,tz_cnt]
    temp_data = pd.DataFrame.from_dict(temporal_dict,orient='index',columns=['dow_prob','dow_cnt','tz_prob','tz_cnt']).reset_index()
    temp_data['user_id'],temp_data['product_id'] = zip(*temp_data['index'])
    del temp_data['index']
    temp_data.to_csv(f'metadata/dow_tz_prob_{suffix}.csv',index=False)

#%%





