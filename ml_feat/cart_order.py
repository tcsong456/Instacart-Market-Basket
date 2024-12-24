# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 20:02:02 2024

@author: congx
"""
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--mode',required=True,choices=[0,1],type=int)
args = parser.parse_args()

data = pd.read_csv('data/orders_info.csv')
df = data[data['reverse_order_number']>args.mode]
cart_order_stats = df.groupby('product_id')['add_to_cart_order'].agg(['min','max','mean','median','std']).add_prefix('cart_order_')
df['order_size'] = df.groupby(['user_id','order_id']).transform('size')
df['cart_order_ratio'] = df['add_to_cart_order'] / df['order_size']
cart_order_ratio_stats = df.groupby('product_id')['cart_order_ratio'].agg(['min','max','mean','median','std']
                                            ).add_prefix('cart_order_ratio_')
cart_order_size = df.groupby('product_id')['order_size'].agg(['min','max','mean','median','std']).add_prefix('order_size_')
cart_order_data = pd.concat([cart_order_stats,cart_order_ratio_stats,cart_order_size],axis=1).reset_index()
suffix = 'test' if args.mode==0 else 'train'
cart_order_data.to_csv(f'metadata/cart_order_data_{suffix}.csv',index=False)


#%%
