# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 22:00:21 2024

@author: congx
"""
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--mode',required=True,choices=[0,1],type=int)
args = parser.parse_args()

suffix = 'test' if args.mode==0 else 'train'
data = pd.read_csv('data/orders_info.csv')
data_tr = data[data['cate']=='train']
data_te = data[data['cate']=='test']
if suffix == 'train':
    df_tr = data_tr[data_tr['reverse_order_number']>0]
    df_te = data_te[data_te['reverse_order_number']>1]
    df = pd.concat([df_tr,df_te])
else:
    df = data_te[data_te['reverse_order_number']>0]
    
user_buy_grp = df.groupby(['user_id','product_id']).size()
user_buy_grp.name = 'cnt'
user_buy_grp = user_buy_grp.reset_index()
user_buy_prods = user_buy_grp.groupby('product_id')['user_id'].size()
user_buy_prods.name = 'total_cnt'
user_buy_prods = user_buy_prods.reset_index()
one_shot_prods = user_buy_grp[user_buy_grp['cnt']==1]
one_shot_prods = one_shot_prods.groupby('product_id')['user_id'].size()
one_shot_prods.name = 'one_shot_cnt'
one_shot_prods = one_shot_prods.reset_index()
user_buy_prods = pd.merge(user_buy_prods,one_shot_prods,on=['product_id'],how='left')
user_buy_prods['one_shot_prob'] = user_buy_prods['one_shot_cnt'] / user_buy_prods['total_cnt']
user_buy_prods = user_buy_prods[['product_id','one_shot_cnt','one_shot_prob']]
user_buy_prods.to_csv(f'metadata/user_one_shot_{suffix}.csv',index=False)


#%%
