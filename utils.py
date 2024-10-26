# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:16:05 2024

@author: congx
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype,is_float_dtype

def load_data(path):
    def decorator(func):
        data_dict = {}
        def wrapper(*args,**kwargs):
            import os
            data_list = os.listdir(path)
            for dt in data_list:
                d_path = os.path.join(path,dt)
                d = pd.read_csv(d_path)
                d_name = dt[:dt.find('.')]
                data = func(d)
                data_dict[d_name] = data
            return data_dict
        return wrapper
    return decorator

@load_data('data/')
def optimize_dtypes(df):
    for col in df.columns:
        col_data = df[col]
        if is_numeric_dtype(col_data):
            if col_data.min() >= np.iinfo(np.int8).min and col_data.max() <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_data.min() >= np.iinfo(np.int16).min and col_data.max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                pass
        elif is_float_dtype(col_data):
            if col_data.min() >= np.finfo(np.float16).min and col_data.max() <= np.finfo()(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo()(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                pass
    return df
            



#%%
import os
s = 'products.csv'
s[:s.find('.')]
