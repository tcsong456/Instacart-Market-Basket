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
    
    def _check_nan(self,x):
        for k,v in x.items():
            if np.isnan(v):
                v = 0
                x[k] = v
        return x
        
    def fake_adjacent_stat(self):
        fake_stats = defaultdict(lambda:defaultdict(int))
        fake_interval_stats = defaultdict(list)
        no_buy_interval = 0
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc='building true adjacent row stat')
        for _,row in rows:
            user,cur_aisles,next_aisles = row['user_id'],row['aisle_id'],row['aisle_next']
            if not isinstance(next_aisles,list):
                no_buy_interval = 0
                continue
            order_interval = row['days_since_prior_order']
            unique_aisles = self.unique_dict[user]
            for aisle in unique_aisles:
                if aisle not in cur_aisles:
                    no_buy_interval += order_interval
                    fake_stats[aisle]['total_cnt'] += 1
                    if aisle in next_aisles:
                        fake_stats[aisle]['cnt'] += 1
                        fake_interval_stats[aisle].append(no_buy_interval)
                        no_buy_interval = 0
                        
        fake_stats = self._convert_to_prob_dict(fake_stats)
        for key,value in fake_interval_stats.items():
            value = self._convert_list_to_dist(value)
            fake_interval_stats[key] = value
        fake_stats = self._check_nan((fake_stats))
        return fake_stats,fake_interval_stats
        
    def build_data(self):
        raise NotImplementedError("subclass must implement function 'build_data'")
    
    def save(self):
        raise NotImplementedError("subclass must implement function 'save'")


#%%


