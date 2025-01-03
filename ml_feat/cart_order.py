# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 20:02:02 2024

@author: congx
"""
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

def add_to_cart_order(df):
    cart_order_stats = df.groupby('product_id')['add_to_cart_order'].agg(['min','max','mean','median','std']).add_prefix('cart_order_')
    df['order_size'] = df.groupby(['user_id','order_id']).transform('size')
    df['cart_order_ratio'] = df['add_to_cart_order'] / df['order_size']
    cart_order_ratio_stats = df.groupby('product_id')['cart_order_ratio'].agg(['min','max','mean','median','std']
                                                ).add_prefix('cart_order_ratio_')
    cart_order_size = df.groupby('product_id')['order_size'].agg(['min','max','mean','median','std']).add_prefix('order_size_')
    cart_order_data = pd.concat([cart_order_stats,cart_order_ratio_stats,cart_order_size],axis=1)
    return cart_order_data

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
        
    cart_order_data = add_to_cart_order(df)
    cart_order_data = cart_order_data.reset_index()

    if suffix == 'train':
        df_tr = data_tr[(data_tr['reverse_order_number']>0)&(data_tr['reverse_order_number']<=5)]
        df_te = data_te[(data_te['reverse_order_number']>1)&(data_te['reverse_order_number']<=6)]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[(data_te['reverse_order_number']>0)&(data_te['reverse_order_number']<=5)]
        
    recent_cart_order_data = add_to_cart_order(df)
    recent_cart_order_data = recent_cart_order_data.add_suffix('_5').reset_index()
    
    cart_order_data = pd.merge(cart_order_data,recent_cart_order_data,how='left',on=['product_id'])
    cart_order_data.to_csv(f'metadata/cart_order_data_{suffix}.csv',index=False)
    

#%%

