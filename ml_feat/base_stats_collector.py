# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:53:16 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.utils import TMP_PATH,optimize_dtypes,pickle_save_load

class BaseStatsCollector:
    def __init__(self,
                 path,
                 attr):
        data = pd.read_csv(path)
        data['counter'] = 1
        self.data = data
        self.col = f'{attr}_id'
        self.attr = attr
    
    def _shift_stats(self,col):
        stat = self.data.groupby(['user_id','order_id','order_number'])[col].apply(lambda x:x.iloc[-1]).reset_index()
        stat = stat.sort_values(['user_id','order_number'])
        del stat['order_number']
        stat = stat.groupby(['user_id'])[col].shift(-1)
        stat.name = col
        return stat
    
    def _replace_nan(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                d[k] = self._replace_nan(v)
            elif isinstance(v,float) and np.isnan(v):
                d[k] = 0
        return d
    
    def _convert_to_prob_dict(self,x):
        df = pd.DataFrame.from_dict({k:dict(v) for k,v in x.items()},orient='index')
        col1,col2 = df.columns
        df = df[col2] / df[col1]
        df = df.to_dict()
        return df
    
    def fake_adjacent_stats(self,min_interval=0,max_interval=31):
        fake_stats = defaultdict(lambda:defaultdict(int))
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc=f'building fake {self.attr} adjacent row stat')
        key = f'absent_{str(min_interval)}_{str(max_interval)}'
        for _,row in rows:
            user,cur_items,next_items = row['user_id'],row[f'{self.attr}_id'],row[f'{self.attr}_next']
            if np.isnan(next_items).any() or next_items == [-1]:
                continue
            order_interval = row['days_since_prior_order']
            for item in self.unique_dict[user]:
                if order_interval >= min_interval and order_interval < max_interval:
                    if item not in cur_items:
                        fake_stats[item][f'total_{key}'] += 1
                        if item in next_items:
                            fake_stats[item][f'out_{key}'] += 1
        fake_stats = self._convert_to_prob_dict(fake_stats)
        fake_stats = self._replace_nan(fake_stats)
        return fake_stats
    
    def _pick_rows(self,agg_path):
        agg_data = pd.read_pickle(agg_path)
        stats = self.stats.reset_index(drop=True)
        indices = stats.reset_index(drop=True).reset_index().groupby('user_id')['index'].max()
        eval_set = agg_data[['user_id','eval_set']].set_index('user_id')
        x = pd.concat([eval_set,indices],axis=1)
        eval_indices = x.apply(lambda x:x['index']-2 if x['eval_set']=='test' else x['index']-1,axis=1)
        train_indices = x.apply(lambda x:x['index']-3 if x['eval_set']=='test' else x['index']-2,axis=1)
        eval_rows = stats.iloc[eval_indices]
        train_rows = stats.iloc[train_indices]
        eval_index = np.array(eval_set=='test').squeeze().tolist()
        eval_rows = eval_rows.loc[eval_index]
        return train_rows,eval_rows
    
    def summary(self,agg_path=''):
        train_rows,eval_rows = self._pick_rows(agg_path)
        eval_data = self._pick_data(eval_rows)
        train_data = self._pick_data(train_rows)
        for d,p in zip([train_data,eval_data],[f'metadata/{self.attr}_prob_eval.npy',f'metadata/{self.attr}_prob_pred.npy']):
            np.save(p,d)
        return train_data,eval_data
    
    def load(self,path=None):
        if path is None:
            load_data = []
            for p in ['true_stats','fake_stats']:
                path = os.path.join(TMP_PATH,f'{self.attr}_{p}.pkl')
                load_data.append(pickle_save_load(path,mode='load'))
            return load_data

        load_data = pickle_save_load(path,mode='load')
        return load_data
    
    @property
    def mapping(self):
        d = {}
        for v in [range(0,8),range(8,23),range(23,31)]:
            start,end = v.start,v.stop
            d.update({i:f'{start}_{end}'for i in v})
        return d
    
    def build_stats(self):
        raise NotImplementedError("subclass must implement function 'build_stats'")
    
    def build_data(self):
        raise NotImplementedError("subclass must implement function 'build_data'")

        
#%%



