# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:15:59 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.utils import TMP_PATH,pickle_save_load,optimize_dtypes
from ml_feat.base_stats_collector import BaseStatsCollector
from collections import defaultdict

class ProductStatsCollector(BaseStatsCollector):
    def __init__(self,
                 path,
                 attr='product',
                 prod_path=''):
        super().__init__(path=path,
                         attr=attr)
        user_unique_prods = self.data.groupby('user_id')['product_id'].apply(set).apply(list)
        self.unique_dict = user_unique_prods.to_dict()
        self.stats = self.build_data()
        self.products = pd.read_csv(prod_path)
    
    def build_data(self):
        path = os.path.join(TMP_PATH,f'{self.attr}_stats_data.pkl')
        try:
            stats = pd.read_pickle(path)
        except FileNotFoundError:
            stats = []
            order_prod = self.data.sort_values(['user_id','order_number'])
            order_prod_ls = order_prod.groupby(['user_id','order_id','order_number'])[self.col].apply(list).reset_index()
            order_prod_ls = order_prod_ls.sort_values(['user_id','order_number'])
            del order_prod_ls['order_number']
            stats.append(order_prod_ls)
            
            order_prod_shift = order_prod_ls.groupby('user_id')[self.col].shift(-1)
            order_prod_shift.name = f'{self.attr}_next'
            stats.append(order_prod_shift)
            
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
            user,days = row['user_id'],row['days_since_prior_order']
            cur_prods = row['product_id']
            prods = self.unique_dict[user]
            if -1 in prods:
                prods.remove(-1)
            day_map = self.mapping[days]
            for prod in prods:
                # key1 = '0_31'
                key2 = day_map
                if prod in cur_prods:
                    s = true_stats
                else:
                     s = fake_stats
                # prob1 = s[key1][prod]
                prob2 = s[key2][prod]
                data.append([user,prod,prob2])
        data = np.array(data).astype(np.float32)
        return data
    
    def true_adjacent_stat(self,min_interval=0,max_interval=31):
        true_stats = defaultdict(lambda:defaultdict(int))
        key = f'{str(min_interval)}_{str(max_interval)}'
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc=f'building true {self.attr} adjacent row stat')
        for _,row in rows:
            cur_prods,next_prods = row[self.col],row[f'{self.attr}_next']
            if np.isnan(next_prods).any() or next_prods == [-1]:
                continue
            order_interval = row['days_since_prior_order']
            for prod in cur_prods:
                if order_interval >= min_interval and order_interval < max_interval:
                    true_stats[prod][f'total_{key}'] += 1
                    if prod in next_prods:
                        true_stats[prod][f'in_{key}'] += 1
        true_stats = self._convert_to_prob_dict(true_stats)
        true_stats = self._replace_nan(true_stats)
        return true_stats
    
    def _fill(self,dic):
        min_prods,max_prods = self.products['product_id'].min(),self.products['product_id'].max()
        all_prods = set(range(min_prods,max_prods+1))
        for key,value in dic.items():
            keys = set(value.keys())
            missing_keys = all_prods - keys
            for k in missing_keys:
                dic[key][k] = 0
        return dic
    
    def build_stats(self):
        intervals = [(0,31),(0,8),(8,23),(23,31)]
        tstats = {f'{s}_{e}':{} for s,e in intervals}
        fstats = {f'{s}_{e}':{} for s,e in intervals}
        for interval in intervals:
            min_interval,max_interval = interval
            true_stats = self.true_adjacent_stat(min_interval=min_interval,max_interval=max_interval)
            key = f'{min_interval}_{max_interval}'
            tstats[key] = true_stats
            fake_stats = self.fake_adjacent_stats(min_interval=min_interval,max_interval=max_interval)
            fstats[key] = fake_stats
        tstats = self._fill(true_stats)
        fstats = self._fill(fstats)
        # return tstats,fstats
        for stat,stat_path in zip([fstats,tstats],[f'{self.attr}_fake_stats',f'{self.attr}_true_stats']):
            path = os.path.join(TMP_PATH,f'{stat_path}.pkl')
            pickle_save_load(path,stat,mode='save')
        return tstats,fstats
            
if __name__ == '__main__':
    product_collector = ProductStatsCollector(path='data/orders_info.csv',
                                              prod_path='data/products.csv')
    # prod_true_stats,prod_fake_satats = product_collector.build_stats()
    prod_prob_eval,prod_prob_pred = product_collector.summary(agg_path='data/tmp/user_product_info.csv')


#%%
# a = prod_prob_eval[:1000]

