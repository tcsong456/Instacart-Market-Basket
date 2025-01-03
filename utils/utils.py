# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:16:05 2024

@author: congx
"""
import torch
import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from time import time
from pandas.api.types import is_integer_dtype,is_float_dtype

TMP_PATH = 'data/tmp'

def optimize_dtypes(df):
    for col in df.columns:
        col_data = df[col]
        if is_integer_dtype(col_data):
            if col_data.min() >= np.iinfo(np.int8).min and col_data.max() <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_data.min() >= np.iinfo(np.int16).min and col_data.max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                pass
        elif is_float_dtype(col_data):
            if col_data.min() >= np.finfo(np.float16).min and col_data.max() <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                pass
    return df

def load_data(path):
    data_dict = {}
    data_list = os.listdir(path)
    for dt in data_list:
        if dt.endswith('.csv'):
            d_path = os.path.join(path,dt)
            d = pd.read_csv(d_path)
            d_name = dt[:dt.find('.')]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for col in d.columns:
                    num_nulls = pd.isnull(d[col]).sum()
                    if num_nulls > 0:
                        d[col].fillna(-1,inplace=True)
                        logger.warning('the column {} of {} dataset has {} number of null values'.format(col,d_name,num_nulls))
            data = optimize_dtypes(d)
            data_dict[d_name] = data
    return data_dict

def _setup_logger():
    logging.basicConfig(
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger('my_logger')
    return logger
logger = _setup_logger()

def split_time_zone(hour):
    if hour > 6 and hour <= 12:
        return 'morning'
    elif hour > 12 and hour <= 18:
        return 'afternoon'
    elif hour > 18 and hour <= 24:
        return 'evening'
    else:
        return 'midnight'
    
def pickle_save_load(path,data=None,mode='save'):
    name = path.split('/')[-1]
    if '.' in name:
        ind = name.find('.')
        name = name[:ind]
    if mode == 'save':
        assert data is not None,'data must be provided when it is in save mode'
        logger.info(f'saving file {name}')
        with open(path,'wb') as f:
            pickle.dump(data,f)
    elif mode == 'load':
        logger.info(f'loading file {name}')
        with open(path,'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise KeyError(f'{mode} is invalid mode')

def pad(inp,max_len):
    padded_len = max_len - inp.shape[0]
    # inp = np.concatenate([inp,np.zeros(padded_len)])
    inp = torch.Tensor(inp) if type(inp)!=torch.Tensor else inp
    inp = torch.cat([inp,torch.zeros(padded_len)])
    return inp

class Timer:
    def __init__(self,
                 precision=0):
        self.message = 'timer starts'
        self.precision = precision
        
    def __enter__(self):
        print(self.message)
        self.start = time()
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        end = time()
        elapsed = end - self.start
        print(f'it took {elapsed:.{self.precision}f} seconds to complete')

def collect_stats(x,agg_func=[]):
    if len(x) == 0:
        return []
    stats = []
    for func in agg_func:
        f = getattr(np,func)
        s = f(x)
        stats.append(s)
    return stats

def df_stats_agg(x,col):
    popup_stats = []
    user_prod_grp = x.groupby('product_id')[col]
    for func in ['min','max','mean','median','std']:
        f = getattr(user_prod_grp,func)
        stat = f()
        stat.name = f'{col}_{func}'
        popup_stats.append(stat)
    popup_stats = pd.concat(popup_stats,axis=1)
    return popup_stats

#%%
