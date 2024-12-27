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
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode] 
    user_prod_popup = popup_data(df)
    
    df = data[(data['reverse_order_number']>args.mode)&(data['reverse_order_number']<=args.mode+5)] 
    user_prod_popup_5 = popup_data(df).add_suffix('_5')
    user_prod_popup = pd.concat([user_prod_popup,user_prod_popup_5],axis=1).reset_index()
    
    suffix = 'test' if args.mode==0 else 'train'
    user_prod_popup.to_csv(f'metadata/user_popup_{suffix}.csv',index=False)
    
    


#%%





