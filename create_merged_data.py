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
from utils.utils import load_data,optimize_dtypes,split_time_zone,logger
from pandas.api.types import is_float_dtype

logger.info('creating the all merged order data')
week_days_map = {0:'Mon',
                 1:'Tue',
                 2:'Wed',
                 3:'Thu',
                 4:'Fri',
                 5:'Sat',
                 6:'Sun'}

data_dict = load_data('data/')
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
orders['time_zone'] = orders['order_hour_of_day'].apply(split_time_zone)
orders['time_zone'] = orders['order_dow_text'] + '-' + orders['time_zone']
le = LabelEncoder()
orders['time_zone'] = le.fit_transform(orders['time_zone'])
del orders['order_dow_text']

for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
    if is_float_dtype(dtype):
        orders[col] = orders[col].astype(np.int32)
orders = optimize_dtypes(orders)

orders.to_csv('data/orders_info.csv',index=False)



#%%
