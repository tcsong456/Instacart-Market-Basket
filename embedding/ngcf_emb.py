# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:19:39 2024

@author: congx
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import optimize_dtypes,load_data
from pandas.api.types import is_float_dtype

def data_process(path):
    data_dict = load_data(path)
    orders = data_dict['orders']
    orders_prior,orders_train = data_dict['order_products__prior'],data_dict['order_products__train']
    products = data_dict['products']
    aisles,departments = data_dict['aisles'],data_dict['departments']
    
    order_products = pd.concat([orders_prior,orders_train])
    del orders_prior,orders_train 
    orders = orders.merge(order_products,how='left',on='order_id').merge(products,\
              how='left',on='product_id').merge(aisles,how='left',on='aisle_id').merge(departments,how='left',on='department_id')
    orders = orders[orders['eval_set']!='test']
    del order_products
    for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
        if is_float_dtype(dtype):
            if pd.isnull(orders[col]).any():
                orders[col].fillna(0,inplace=True)
            orders[col] = orders[col].astype(np.int64)
    orders = optimize_dtypes(orders)
    
    order_products = orders.groupby(['user_id','order_id'])['product_id'].apply(list).reset_index()
    rows = tqdm(order_products.iterrows(),total=order_products.shape[0])
    coexist_dict = {}
    for _,row in rows:
        unique_items = row['product_id']
        for item1 in unique_items:
            for item2 in unique_items:
                if item1 != item2 and (item1,item2) not in coexist_dict:
                    coexist_dict[(item1,item2)] = 1
    item_item = pd.DataFrame(list(coexist_dict.items()),columns=['product_pair','value'])
    item_item = pd.DataFrame(item_item['product_pair'].tolist())
    
    user_item = orders.groupby('user_id')['product_id'].apply(set).apply(list).reset_index()
    user_item = user_item.explode('product_id')
    return np.array(user_item).astype(np.int32),np.array(item_item).astype(np.int32)

def construct_graph(user_item,item_item):
    total_users = user_item[:,0].max()
    toal_items = user_item[:,1].max()
    users = [i for i in range(total_users)]
    items = [i for i in range(total_items)]
    
    link_dict = {
                  ('user','user_self_link','user'):(users,users),
                  ('item','item_self_link','item'):(items,items),
                  
                    }

if __name__ == '__main__':
    user_item_interaction,item_item_interaction = data_process('data/')

#%%
import gc
user_item_interaction[:,0].tolist()






#%%
# tmp = orders[['order_id','product_id']]
np.array(user_item).astype(np.int32)