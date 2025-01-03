# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:50:08 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd
from utils.utils import df_stats_agg

def popup_data(df):
    user_prod = df.groupby(['user_id','product_id']).size()
    user_prod.name = 'popup_cnt'
    user_prod = user_prod.reset_index()
    user_size = df.drop_duplicates(['user_id','order_id']).groupby(['user_id']).size()
    user_size.name = 'total'
    user_size = user_size.reset_index()
    user_prod = pd.merge(user_prod,user_size,how='left',on=['user_id'])
    user_prod['popup_ratio'] = user_prod['popup_cnt'] / user_prod['total']
    user_prod.set_index(['user_id','product_id'],inplace=True)
    del user_prod['total']
    
    popup_cnt_stats = df_stats_agg(user_prod,'popup_cnt')
    popup_ratio_stats = df_stats_agg(user_prod,'popup_ratio')
    popup_data = pd.concat([popup_cnt_stats,popup_ratio_stats],axis=1)
    return popup_data

if __name__ == '__main__':
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
    user_prod_popup = popup_data(df)
    
    if suffix == 'train':
        df_tr = data_tr[(data_tr['reverse_order_number']>0)&(data_tr['reverse_order_number']<=5)]
        df_te = data_te[(data_te['reverse_order_number']>1)&(data_te['reverse_order_number']<=6)]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[(data_te['reverse_order_number']>0)&(data_te['reverse_order_number']<=5)]
    user_prod_popup_5 = popup_data(df).add_suffix('_5')
    user_prod_popup = pd.concat([user_prod_popup,user_prod_popup_5],axis=1).reset_index()
    
    user_prod_popup.to_csv(f'metadata/user_popup_{suffix}.csv',index=False)
    
    


#%%





