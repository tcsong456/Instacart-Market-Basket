# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:59:28 2024

@author: congx
"""
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def build_withinN_order(df):
    cnt_dict = defaultdict(int)
    within_n_dict = defaultdict(lambda:defaultdict(int))
    rows = tqdm(df.iterrows(),total=df.shape[0],desc='building within N order data')
    for _,row in rows:
        prod = row['product_id']
        order_numbers = np.array(row['order_number'])
        if len(order_numbers) == 1:
            continue
        for orn in order_numbers:
            orn_start,orn_end = order_numbers[:-1],order_numbers[1:]
            intervals = orn_end - orn_start
            for intl in intervals:
                intl = 5 if intl > 5 else intl
                within_n_dict[prod][intl] += 1
                cnt_dict[prod] += 1
    interval_prods = pd.DataFrame.from_dict(within_n_dict,orient='index').add_prefix('interval_').reset_index().rename(columns={'index':'product_id'})
    interval_cnt = pd.DataFrame.from_dict(cnt_dict,orient='index',columns=['interval_cnt'])
    interval_cnt = interval_cnt.reset_index().rename(columns={'index':'product_id'})
    interval_prods = pd.merge(interval_prods,interval_cnt,how='left',on=['product_id'])
    columns = interval_prods.columns.tolist()
    columns = [col for col in columns if bool(re.search(r'\d',col))]
    interval_probs = np.array(interval_prods[columns]) / np.array(interval_prods['interval_cnt']).reshape(-1,1)
    interval_probs = pd.DataFrame(interval_probs,columns=columns).add_suffix('_prob')
    interval_prods = pd.concat([interval_prods,interval_probs],axis=1)
    
    interval_n_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    rows = tqdm(interval_prods.iterrows(),total=interval_prods.shape[0],desc='building within N order dict')
    columns = interval_prods.columns.tolist()
    for _,row in rows:
        for ind in list(map(str,np.arange(1,6))):
            cols = [c for c in columns if ind in c]
            cnt,prob = cols
            cnt,prob = row[cnt],row[prob]
            prod = row['product_id']
            interval_n_dict[prod][int(ind)]['cnt'] = cnt
            interval_n_dict[prod][int(ind)]['prob'] = prob
    return interval_n_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=[0,1],type=int)
    args = parser.parse_args()
    
    data = pd.read_csv('data/orders_info.csv')
    df = data[data['reverse_order_number']>args.mode]
    user_prod_orn = df.groupby(['user_id','product_id'])['order_number'].apply(list).reset_index()
    interval_n_dict = build_withinN_order(user_prod_orn)
    
    interval_stats = {}
    prod_orn_max = df.groupby(['user_id','product_id'])['reverse_order_number'].min().reset_index()
    rows = tqdm(prod_orn_max.iterrows(),total=prod_orn_max.shape[0],desc='collecting n order stat',leave=False)
    for _,row in rows:
        user,prod = row['user_id'],row['product_id']
        rev_orn = row['reverse_order_number']
        key = rev_orn - args.mode
        key = key if key <= 5 else 5
        intl_cnt = interval_n_dict[prod][key]['cnt']
        intl_prob = interval_n_dict[prod][key]['prob']
        interval_stats[(user,prod)] = [intl_cnt,intl_prob]

    interval_stats = pd.DataFrame.from_dict(interval_stats,orient='index',columns=['cnt','prob']).add_prefix('n_order_').reset_index()
    interval_stats['user_id'],interval_stats['product_id'] = zip(*interval_stats['index'])
    del interval_stats['index']
    suffix = 'test' if args.mode==0 else 'train'
    interval_stats.to_csv(f'metadata/N_order_{suffix}.csv',index=False)
    
    
    


#%%