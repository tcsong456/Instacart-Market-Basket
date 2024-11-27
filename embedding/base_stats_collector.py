# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:46:26 2024

@author: congx
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter,defaultdict

class BaseStatsCollector:
    def __init__(self,
                 data):
        self.data = data
        self.data['counter'] = 1
        
    def _convert_to_prob_dict(self,x):
        df = pd.DataFrame.from_dict({k:dict(v) for k,v in x.items()},orient='index')
        col1,col2 = df.columns
        df = df[col2] / df[col1]
        df = df.to_dict()
        return df
    
    def _convert_list_to_dist(self,x):
        counter = Counter(x)
        df = pd.DataFrame.from_dict(counter,orient='index')
        df /= df.sum()
        df = df.to_dict()
        return df[0]
    
    def _replace_nan(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                d[k] = self._replace_nan(v)
            elif isinstance(v,float) and np.isnan(v):
                d[k] = 0
        return d
        
    def fake_adjacent_stat(self):
        all_aisles = self.data['aisle_id'].unique()
        all_aisles = all_aisles[all_aisles!=-1]
        fake_stats = defaultdict(lambda:defaultdict(int))
        fake_interval_stats = defaultdict(int)
        fake_ls = defaultdict(list)
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc='building fake adjacent row stat')
        for _,row in rows:
            user,cur_aisles,next_aisles = row['user_id'],row['aisle_id'],row['aisle_next']
            if np.isnan(next_aisles).any() or next_aisles == [-1]:
                for k in all_aisles:
                    fake_interval_stats[k] = 0
                continue
            order_interval = row['days_since_prior_order']
            unique_aisles = self.unique_dict[user]
            for aisle in unique_aisles:
                if aisle not in cur_aisles:
                    fake_interval_stats[aisle] += order_interval
                    fake_stats[aisle]['total_cnt'] += 1
                    if aisle in next_aisles:
                        fake_stats[aisle]['cnt'] += 1
                        fake_ls[aisle].append(fake_interval_stats[aisle])
                        fake_interval_stats[aisle] = 0
                        
        fake_stats = self._convert_to_prob_dict(fake_stats)
        for key,value in fake_ls.items():
            value = self._convert_list_to_dist(value)
            fake_ls[key] = value
        fake_stats = self._replace_nan((fake_stats))
        return fake_stats,fake_ls
    
    @property
    def mapping(self):
        d = {}
        for v in [range(0,8),range(8,23),range(23,31)]:
            start,end = v.start,v.stop
            d.update({i:f'{start}_{end}'for i in v})
        return d
        
    def build_data(self):
        raise NotImplementedError("subclass must implement function 'build_data'")
    
    def save(self):
        raise NotImplementedError("subclass must implement function 'save'")


#%%


        