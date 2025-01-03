# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:28:23 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.utils import df_stats_agg

def build_streaks(df):
    streak_dict = {}
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building streaks for user prod pair')
    for _,row in rows:
        user,product = row['user_id'],row['product_id']
        order_number = row['order_number']
        max_orn = row['max_orn']
        streaks = np.ones_like(order_number)
        for i in range(len(order_number)-1):
            if order_number[i+1] - order_number[i] == 1:
                streaks[i+1] += streaks[i]
        added_len = 0
        for i in range(len(order_number)):
            if i == 0 and order_number[i] > 1:
                gap = order_number[i] - 1
                insert_list = np.arange(0,-gap,-1)
                streaks = np.insert(streaks,i,insert_list)
                added_len += len(insert_list)
            if i < len(order_number) - 1:
                gap = order_number[i+1] - (order_number[i] + 1)
                insert_list = np.arange(0,-gap,-1)
                streaks = np.insert(streaks,i+1+added_len,insert_list)
                added_len += len(insert_list)
        if order_number[-1] < max_orn:
            gap = max_orn - order_number[-1]
            insert_list = np.arange(0,-gap,-1)
            streaks = np.insert(streaks,len(order_number)+added_len,insert_list)
        streak_dict[(user,product)] = streaks.tolist()
    
    streak = pd.DataFrame({'key':streak_dict.keys(),
                            'streak':[list(value) for value in streak_dict.values()]})
    streak['user_id'],streak['product_id'] = zip(*streak['key'])
    del streak['key']
    streak = streak.explode('streak')
    return streak
        
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    suffix = 'test' if args.mode==0 else 'train'
    data = pd.read_csv('data/orders_info.csv')
    data_tr = data[data['cate']=='train']
    data_te = data[data['cate']=='test']
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']>0]
        df_te = data_te[data_te['reverse_order_number']>1]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']>0]
        
    df = df.sort_values(['user_id','order_number'])
    user_prod_orn = df.groupby(['user_id','product_id'])['order_number'].apply(list).reset_index()
    user_orn_max = df.groupby(['user_id'])['order_number'].max()
    user_orn_max.name = 'max_orn'
    user_prod_orn = user_prod_orn.merge(user_orn_max,how='left',on=['user_id'])
    streak = build_streaks(user_prod_orn)
    streak_stat = df_stats_agg(streak,'streak').reset_index()
    streak_stat.to_csv(f'metadata/streak_{suffix}.csv',index=False)



#%%


