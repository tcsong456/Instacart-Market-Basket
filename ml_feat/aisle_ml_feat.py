# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:31:42 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.utils import TMP_PATH,pickle_save_load,optimize_dtypes
from ml_feat.base_stats_collector import BaseStatsCollector
from collections import defaultdict

class AisleStatsCollector(BaseStatsCollector):
    def __init__(self,
                 path,
                 attr='aisle'):
        super().__init__(path=path,
                         attr=attr)
        user_unique_aisles = self.data.groupby('user_id')['aisle_id'].apply(set).apply(list)
        self.unique_dict = user_unique_aisles.to_dict()
        self.stats = self.build_data()
    
    def build_data(self):
        path = os.path.join(TMP_PATH,f'{self.attr}_stats_data.pkl')
        try:
            stats = pd.read_pickle(path)
        except FileNotFoundError:
            stats = []
            order_aisle = self.data.groupby(['user_id','order_id','order_number',self.col])['counter'].sum().reset_index()
            order_aisle = order_aisle.sort_values(['user_id','order_number'])
            order_aisle_ls = order_aisle.groupby(['user_id','order_id','order_number'])[self.col].apply(list).reset_index()
            order_aisle_ls = order_aisle_ls.sort_values(['user_id','order_number'])
            del order_aisle_ls['order_number']
            stats.append(order_aisle_ls)
            
            order_aisle_shift = order_aisle_ls.groupby('user_id')[self.col].shift(-1)
            order_aisle_shift.name = f'{self.attr}_next'
            stats.append(order_aisle_shift)
            
            order_aisle_cnt = order_aisle.groupby(['user_id','order_id','order_number'])['counter'].apply(list).reset_index()
            order_aisle_cnt = order_aisle_cnt.sort_values(['user_id','order_number'])
            del order_aisle_cnt['order_number']
            stats.append(order_aisle_cnt[['counter']])
            
            for col in ['order_dow','days_since_prior_order']:
                s = self._shift_stats(col)
                stats.append(s)
            stats = pd.concat(stats,axis=1)
            stats = optimize_dtypes(stats)
            stats.to_pickle(path)
        return stats
    
    def _pick_data(self,rows):
        data = []
        # true_stats = pickle_save_load(f'data/tmp/{self.attr}_true_stats.pkl',mode='load')
        # fake_stats = pickle_save_load(f'data/tmp/{self.attr}_fake_stats.pkl',mode='load')
        true_stats,fake_stats = self.build_stats()
        rows = tqdm(rows.iterrows(),total=rows.shape[0],desc=f'pickle data for {self.attr} rows')
        for _,row in rows:
            user,days,cnts = row['user_id'],row['days_since_prior_order'],row['counter']
            cur_aisles = row['aisle_id']
            aisles = self.unique_dict[user]
            if -1 in aisles:
                aisles.remove(-1)
            day_map = self.mapping[days]
            for aisle in aisles:
                if aisle in cur_aisles:
                    index = cur_aisles.index(aisle)
                    cnt = cnts[index]
                    cnt = cnt if cnt <= 4 else 4
                    cnt = str(cnt)
                    # key1 = f'cnt_{cnt}_0_31'
                    key2 = f'cnt_{cnt}_{day_map}'
                    # prob1 = true_stats[cnt][key1][aisle]
                    prob2 = true_stats[cnt][key2][aisle]
                else:
                     # key1 = '0_31'
                     key2 = day_map
                     # prob1 = fake_stats[key1][aisle]
                     prob2 = fake_stats[key2][aisle]
                data.append([user,aisle,prob2])
        data = np.array(data).astype(np.float32)
        return data
    
    def true_adjacent_stat(self,valid_cnt=1,min_interval=0,max_interval=31,comparator='equal'):
        operators = {'equal':operator.eq,
                     'greater_equal':operator.ge}
        op = operators[comparator]
        true_stats = defaultdict(lambda:defaultdict(int))
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc=f'building true {self.attr} adjacent row stat')
        key = f'{str(valid_cnt)}_{str(min_interval)}_{str(max_interval)}'
        for _,row in rows:
            cur_aisles,next_aisles,cnts = row[self.col],row[f'{self.attr}_next'],row['counter']
            if np.isnan(next_aisles).any() or next_aisles == [-1]:
                continue
            order_interval = row['days_since_prior_order']
            for aisle in cur_aisles:
                index = cur_aisles.index(aisle)
                cnt = cnts[index]
                if op(cnt,valid_cnt) and order_interval >= min_interval and order_interval < max_interval:
                    true_stats[aisle][f'total_{key}'] += 1
                    if aisle in next_aisles:
                        true_stats[aisle][f'in_{key}'] += 1
                            
        true_stats = self._convert_to_prob_dict(true_stats)
        true_stats = self._replace_nan(true_stats)
        return true_stats
    
    def build_stats(self):
        tstats = {str(k):{} for k in [1,2,3,4]}
        # (0,31),
        intervals = [(0,8),(8,23),(23,31)]
        fstats = {f'{s}_{e}':{} for s,e in intervals}
        for interval in intervals:
            min_interval,max_interval = interval
            for cnt in [1,2,3,4]:
                params = {'valid_cnt':cnt,'min_interval':min_interval,'max_interval':max_interval}
                if cnt <= 3:
                    params.update({'comparator':'equal'}) 
                else:
                    params.update({'comparator':'greater_equal'})
                true_stats = self.true_adjacent_stat(**params)
                key = f'cnt_{str(cnt)}_{str(min_interval)}_{str(max_interval)}'
                tstats[str(cnt)][key] = true_stats
            
            fake_stats = self.fake_adjacent_stats(min_interval=min_interval,max_interval=max_interval)
            key = f'{min_interval}_{max_interval}'
            fstats[key] = fake_stats
        # return tstats,fstats
        for stat,stat_path in zip([fstats,tstats],[f'{self.attr}_fake_stats',f'{self.attr}_true_stats']):
            path = os.path.join(TMP_PATH,f'{stat_path}.pkl')
            pickle_save_load(path,stat,mode='save')
        return tstats,fstats
    

if __name__ == '__main__':
    aisle_collector = AisleStatsCollector(path='data/orders_info.csv')
    # aisle_collector.build_stats()
    train_data,eval_data = aisle_collector.summary(agg_path='data/tmp/user_product_info.csv')
    
    

#%%
# aisle_true_stats = pickle_save_load('data/tmp/aisle_true_stats.pkl',mode='load')
# aisle_fake_stats = pickle_save_load('data/tmp/aisle_fake_stats.pkl',mode='load')
# import torch
# checkpoint = torch.load('checkpoint/AisleTemporalNet_best_checkpoint.pth')
# aisle_collector.mapping
# aisle_true_stats.keys()
# x = stats.iloc[:1000]

