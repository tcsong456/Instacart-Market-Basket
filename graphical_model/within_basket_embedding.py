# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:58:07 2024

@author: congx
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils import load_data
from torch.utils.data import Dataset,DataLoader
from torch import nn
from pandas.api.types import is_integer_dtype,is_float_dtype

class DNNSTP(nn.Module):
    def __init__(self,
                 n_items,
                 emb_dim):
        self.item_embedding = nn.Embedding(n_items,emb_dim)

class DnnstpSet(Dataset):
    def __init__(self,
                 data_path,
                 item_embedding):
        data_dict = load_data('data/')
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
        
        from utils import optimize_dtypes
        for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
            if is_float_dtype(dtype):
                if pd.isnull(orders[col]).any():
                    orders[col].fillna(0,inplace=True)
                orders[col] = orders[col].astype(np.int64)
        orders = optimize_dtypes(orders)
        user_order_products = orders.groupby(['user_id','order_id','order_number'])['product_id'].apply(list)
        user_order_products = user_order_products.reset_index().sort_values(['user_id','order_number'])
        user_products = user_order_products.groupby(['user_id'])['product_id'].apply(list)
        del user_order_products



#%%
def aa():

    return user_products


#%%
# user_products = aa()
x = user_products.iloc[:100]



