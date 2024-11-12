# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:16:05 2024

@author: congx
"""
import os
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from time import time
from torch import nn
from torch.nn import functional as F
from pandas.api.types import is_integer_dtype,is_float_dtype

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
            elif col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo()(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                pass
    return df

def load_data(path):
    data_dict = {}
    data_list = os.listdir(path)
    for dt in data_list:
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
    
# def get_label(dts,num_classes):
#     ohs = []
#     for dt in dts:
#         dt = torch.tensor(dt)
#         oh = F.one_hot(dt,num_classes=num_classes)
#         oh_basket,_ = torch.max(oh,dim=0)
#         ohs.append(oh_basket)
#     label = torch.stack(ohs,dim=0)
#     return label

def pad(inp,max_len):
    padded_len = max_len - inp.shape[0]
    inp = list(inp) + [0] * padded_len
    return inp

class SeqLogLoss(nn.Module):
    def __init__(self,
                 eps=1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self,y_pred,y_true,lengths):
        y_pred = torch.sigmoid(y_pred)
        logloss = y_true * torch.log(y_pred + self.eps) + (1 - y_true) * torch.log(1 - y_pred + self.eps)
        mask = torch.zeros(y_true.shape).cuda()
        for idx,length in enumerate(lengths):
            mask[idx,:length-1] = 1
        total = sum(lengths) - len(lengths)
        loss = -(logloss * mask).sum() / total
        return loss

class Timer:
    def __init__(self,
                 precision):
        self.message = 'timer starts'
        self.precision = precision
        
    def __enter__(self):
        print(self.message)
        self.start = time()
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        end = time()
        elapsed = end - self.start
        print(f'it took {elapsed:.{self.precision}f} seconds to complete')

#%%



