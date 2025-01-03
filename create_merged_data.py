# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:09:19 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.utils import load_data,optimize_dtypes,split_time_zone,logger,TMP_PATH
from pandas.api.types import is_float_dtype

def create_orders_info(path):
    logger.info('creating the all merged order data')
    week_days_map = {0:'Mon',
                     1:'Tue',
                     2:'Wed',
                     3:'Thu',
                     4:'Fri',
                     5:'Sat',
                     6:'Sun'}
    
    data_dict = load_data(path)
    orders = data_dict['orders']
    orders_prior,orders_train = data_dict['order_products__prior'],data_dict['order_products__train']
    products = data_dict['products']
    aisles,departments = data_dict['aisles'],data_dict['departments']
    
    order_products = pd.concat([orders_prior,orders_train])
    del orders_prior,orders_train 
    orders = orders.merge(order_products,how='left',on='order_id').merge(products,\
              how='left',on='product_id').merge(aisles,how='left',on='aisle_id').merge(departments,how='left',on='department_id')
    orders.fillna(-1,inplace=True)
    del order_products
    
    orders['order_dow_text'] = orders['order_dow'].map(week_days_map)
    orders['hour_zone'] = orders['order_hour_of_day'].apply(split_time_zone)
    orders['time_zone'] = orders['order_dow_text'] + '-' + orders['hour_zone']
    le = LabelEncoder()
    orders['time_zone'] = le.fit_transform(orders['time_zone'])
    del orders['order_dow_text']
    
    orders['max_order_number'] = orders.groupby('user_id')['order_number'].transform(np.max)
    orders['reverse_order_number'] = orders['max_order_number'] - orders['order_number']
    del orders['max_order_number']
    
    orders['cate'] = orders.groupby(['user_id'])['eval_set'].transform(lambda x:x.iloc[-1])
    
    for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
        if is_float_dtype(dtype):
            orders[col] = orders[col].astype(np.int32)
    orders = optimize_dtypes(orders)
    
    orders.to_csv('data/orders_info.csv',index=False)
    

def data_processing(data,save=False):
    os.makedirs(TMP_PATH,exist_ok=True)
    path = os.path.join(TMP_PATH,'user_product_info.csv')
    try:
        r = pd.read_pickle(path)
    except FileNotFoundError:
        logger.info('building user data')
        data = data.sort_values(['user_id','order_number','add_to_cart_order'])
        grouped_data = []
        
        for col in ['product_id','reordered','aisle_id','department_id']:
            x = data.groupby(['user_id','order_id','order_number'])[col].apply(list).map(lambda x:'_'.join(map(str,x))).reset_index()
            x = x.sort_values(['user_id','order_number']).groupby(['user_id'])[col].apply(list)
            grouped_data.append(x)
        r = pd.concat(grouped_data,axis=1)
        
        grouped_attr = []
        for col in ['order_hour_of_day','order_dow','days_since_prior_order','time_zone']:
            x = data.groupby(['user_id','order_id','order_number']).apply(lambda x:x[col].iloc[0])
            x.name = col
            x = x.reset_index().sort_values(['user_id','order_number']).groupby('user_id')[col].apply(list)
            grouped_attr.append(x)
        x = pd.concat(grouped_attr,axis=1)
        
        z = data.groupby('user_id').apply(lambda x:x['eval_set'].iloc[-1])
        z.name = 'eval_set'
        
        r = r.merge(x,how='left',on='user_id').merge(z,how='left',on='user_id').reset_index()
        if save:
            r.to_pickle(path)
    return r

if __name__ == '__main__':
    create_orders_info('data/')

#%%
