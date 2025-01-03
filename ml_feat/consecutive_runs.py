# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:49:57 2024

@author: congx
"""
import gc
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def convertion(df,is_prob=True):
    d = pd.DataFrame.from_dict(df)
    d = d[d['total']>=10]
    d.fillna(0,inplace=True)
    if is_prob:
        d['prob'] = d['cnt'] / d['total']
        ddict = d['prob'].to_dict()
        return ddict
    ddict = d['cnt'].to_dict()
    return ddict

def x_to_x(data,unq_dict):
    true_cnt_stat = defaultdict(lambda:defaultdict(int))
    fake_cnt_stat = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 1 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods = row['product_id'],row['shift_1_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                true_cnt_stat['total'][prod] += 1
                if prod in s1_prods:
                    true_cnt_stat['cnt'][prod] += 1
            else:
                fake_cnt_stat['total'][prod] += 1
                if prod in s1_prods:
                    fake_cnt_stat['cnt'][prod] += 1
    
    true_prob_stat = convertion(true_cnt_stat)
    fake_prob_stat = convertion(fake_cnt_stat)
    true_cnt_stat = convertion(true_cnt_stat,False)
    fake_cnt_stat = convertion(fake_cnt_stat,False)
    return true_cnt_stat,true_prob_stat,fake_cnt_stat,fake_prob_stat

def xx_to_x(data,unq_dict):
    stat_cnt_111 = defaultdict(lambda:defaultdict(int))
    stat_cnt_101 = defaultdict(lambda:defaultdict(int))
    stat_cnt_011 = defaultdict(lambda:defaultdict(int))
    stat_cnt_001 = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 2 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods,s2_prods = row['product_id'],row['shift_1_products'],row['shift_2_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                if prod in s1_prods:
                    stat_cnt_111['total'][prod] += 1
                    if prod in s2_prods:
                        stat_cnt_111['cnt'][prod] += 1
                else:
                    stat_cnt_101['total'][prod] += 1
                    if prod in s2_prods:
                        stat_cnt_101['cnt'][prod] += 1
            else:
                if prod in s1_prods:
                    stat_cnt_011['total'][prod] += 1
                    if prod in s2_prods:
                        stat_cnt_011['cnt'][prod] += 1
                else:
                    stat_cnt_001['total'][prod] += 1
                    if prod in s2_prods:
                        stat_cnt_001['cnt'][prod] += 1
    
    stat_prob_111 = convertion(stat_cnt_111)
    stat_prob_101 = convertion(stat_cnt_101)
    stat_prob_011 = convertion(stat_cnt_011)
    stat_prob_001 = convertion(stat_cnt_001)
    stat_cnt_111 = convertion(stat_cnt_111,False)
    stat_cnt_101 = convertion(stat_cnt_101,False)
    stat_cnt_011 = convertion(stat_cnt_011,False)
    stat_cnt_001 = convertion(stat_cnt_001,False)
    return stat_cnt_111,stat_prob_111,stat_cnt_101,stat_prob_101,stat_cnt_011,stat_prob_011,stat_cnt_001,stat_prob_001

def xxx_to_x(data,unq_dict):
    stat_cnt_1111 = defaultdict(lambda:defaultdict(int))
    stat_cnt_1101 = defaultdict(lambda:defaultdict(int))
    stat_cnt_1001 = defaultdict(lambda:defaultdict(int))
    stat_cnt_1011 = defaultdict(lambda:defaultdict(int))
    stat_cnt_0111 = defaultdict(lambda:defaultdict(int))
    stat_cnt_0101 = defaultdict(lambda:defaultdict(int))
    stat_cnt_0001 = defaultdict(lambda:defaultdict(int))
    stat_cnt_0011 = defaultdict(lambda:defaultdict(int))
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='building adjacent 3 data')
    for _,row in rows:
        user = row['user_id']
        prods,s1_prods,s2_prods,s3_prods = row['product_id'],row['shift_1_products'],row['shift_2_products'],row['shift_3_products']
        unique_prods = unq_dict[user]
        for prod in unique_prods:
            if prod in prods:
                if prod in s1_prods:
                    if prod in s2_prods:
                        stat_cnt_1111['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_1111['cnt'][prod] += 1
                    else:
                        stat_cnt_1101['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_1101['cnt'][prod] += 1
                else:
                    if prod in s2_prods:
                        stat_cnt_1011['total'][prod] +=1
                        if prod in s3_prods:
                            stat_cnt_1011['cnt'][prod] += 1
                    else:
                        stat_cnt_1001['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_1001['cnt'][prod] += 1
            else:
                if prod in s1_prods:
                    if prod in s2_prods:
                        stat_cnt_0111['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_0111['cnt'][prod] += 1
                    else:
                        stat_cnt_0101['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_0101['cnt'][prod] += 1
                else:
                    if prod in s2_prods:
                        stat_cnt_0011['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_0011['cnt'][prod] += 1
                    else:
                        stat_cnt_0001['total'][prod] += 1
                        if prod in s3_prods:
                            stat_cnt_0001['cnt'][prod] += 1
     
    stat_prob_1111 = convertion(stat_cnt_1111)
    stat_prob_1101 = convertion(stat_cnt_1101)
    stat_prob_1001 = convertion(stat_cnt_1001)
    stat_prob_1011 = convertion(stat_cnt_1011)
    stat_prob_0111 = convertion(stat_cnt_0111)
    stat_prob_0101 = convertion(stat_cnt_0101)
    stat_prob_0001 = convertion(stat_cnt_0001)
    stat_prob_0011 = convertion(stat_cnt_0011)
    
    stat_cnt_1111 = convertion(stat_cnt_1111,False)
    stat_cnt_1101 = convertion(stat_cnt_1101,False)
    stat_cnt_1001 = convertion(stat_cnt_1001,False)
    stat_cnt_1011 = convertion(stat_cnt_1011,False)
    stat_cnt_0111 = convertion(stat_cnt_0111,False)
    stat_cnt_0101 = convertion(stat_cnt_0101,False)
    stat_cnt_0001 = convertion(stat_cnt_0001,False)
    stat_cnt_0011 = convertion(stat_cnt_0011,False)
    return stat_cnt_1111,stat_cnt_1101,stat_cnt_1001,stat_cnt_1011,stat_cnt_0111,stat_cnt_0101,stat_cnt_0001,stat_cnt_0011,\
           stat_prob_1111,stat_prob_1101,stat_prob_1001,stat_prob_1011,stat_prob_0111,stat_prob_0101,stat_prob_0001,stat_prob_0011
                            
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
    unique_data_dict = df.groupby('user_id')['product_id'].apply(set).to_dict()
    
    user_prods = df.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    user_prods = user_prods.sort_values(['user_id','order_number'])
    prods_shift_1 = user_prods.groupby('user_id')['product_id'].shift(-1)
    prods_shift_1.name = 'shift_1_products'
    adjacent_basket_1 = pd.concat([user_prods,prods_shift_1],axis=1)
    adjacent_basket_1.dropna(axis=0,inplace=True)
    stat_cnt_11,stat_prob_11,stat_cnt_01,stat_prob_01 = x_to_x(adjacent_basket_1,unique_data_dict)
    
    prods_shift_2 = user_prods.groupby('user_id')['product_id'].shift(-2)
    prods_shift_2.name = 'shift_2_products'
    adjacent_basket_2 = pd.concat([user_prods,prods_shift_1,prods_shift_2],axis=1)
    adjacent_basket_2.dropna(axis=0,inplace=True)
    stat_cnt_111,stat_prob_111,stat_cnt_101,stat_prob_101,stat_cnt_011,stat_prob_011,\
    stat_cnt_001,stat_prob_001 = xx_to_x(adjacent_basket_2,unique_data_dict)
    
    prods_shift_3 = user_prods.groupby('user_id')['product_id'].shift(-3)
    prods_shift_3.name = 'shift_3_products'
    adjacent_basket_3 = pd.concat([user_prods,prods_shift_1,prods_shift_2,prods_shift_3],axis=1)
    adjacent_basket_3.dropna(axis=0,inplace=True)
    stat_cnt_1111,stat_cnt_1101,stat_cnt_1001,stat_cnt_1011,stat_cnt_0111,stat_cnt_0101,stat_cnt_0001,stat_cnt_0011,\
    stat_prob_1111,stat_prob_1101,stat_prob_1001,stat_prob_1011,stat_prob_0111,stat_prob_0101,stat_prob_0001,\
    stat_prob_0011 = xxx_to_x(adjacent_basket_3,unique_data_dict)
    
    probs = dict()
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']==1]
        df_te = data_te[data_te['reverse_order_number']==2]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']==1]
    target_1_grp = df.groupby('user_id')['product_id'].apply(list).reset_index()
    rows = tqdm(target_1_grp.iterrows(),total=target_1_grp.shape[0],desc='collecting stats for two consecutive run of products')
    for _,row in rows:
        user,products = row['user_id'],row['product_id']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in products:
                prob = stat_prob_11.get(prod,np.nan)
                cnt = stat_cnt_11.get(prod,np.nan)
            else:
                prob = stat_prob_01.get(prod,np.nan)
                cnt = stat_cnt_01.get(prod,np.nan)
            probs[(user,prod)] = [prob,cnt]

    probs_xx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xx','cnt_xx']).reset_index()
    probs_xx['user_id'],probs_xx['product_id'] = zip(*probs_xx['index'])
    del probs_xx['index']

    probs = dict()
    df = data[data['reverse_order_number']==args.mode+1]
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']==2]
        df_te = data_te[data_te['reverse_order_number']==3]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']==2]
    target_2_grp = df.groupby('user_id')['product_id'].apply(list).reset_index().rename(columns={'product_id':'shift_1_products'})
    target_grp = pd.merge(target_2_grp,target_1_grp,how='left',on=['user_id'])
    rows = tqdm(target_grp.iterrows(),total=target_grp.shape[0],desc='collecting stats for three consecutive run of products')
    for _,row in rows:
        user,products,s1_products = row['user_id'],row['product_id'],row['shift_1_products']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in s1_products:
                if prod in products:
                    prob = stat_prob_111.get(prod,np.nan)
                    cnt = stat_cnt_111.get(prod,np.nan)
                else:
                    prob = stat_prob_101.get(prod,np.nan)
                    cnt = stat_cnt_101.get(prod,np.nan)
            else:
                if prod in products:
                    prob = stat_prob_011.get(prod,np.nan)
                    cnt = stat_cnt_011.get(prod,np.nan)
                else:
                    prob = stat_prob_001.get(prod,np.nan)
                    cnt = stat_cnt_001.get(prod,np.nan)
            probs[(user,prod)] = [prob,cnt]
    probs_xxx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xxx','cnt_xxx']).reset_index()
    probs_xxx['user_id'],probs_xxx['product_id'] = zip(*probs_xxx['index'])
    del probs_xxx['index']
    
    probs = dict()
    df = data[data['reverse_order_number']==args.mode+1]
    if suffix == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']==3]
        df_te = data_te[data_te['reverse_order_number']==4]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']==3]
    target_3_grp = df.groupby('user_id')['product_id'].apply(list).reset_index().rename(columns={'product_id':'shift_2_products'})
    target_grp = target_3_grp.merge(target_2_grp,how='left',on=['user_id']).merge(target_1_grp,how='left',on=['user_id'])
    target_grp.dropna(axis=0,inplace=True)
    rows = tqdm(target_grp.iterrows(),total=target_grp.shape[0],desc='collecting stats for three consecutive run of products')
    for _,row in rows:
        user,products,s1_products,s2_products = row['user_id'],row['product_id'],row['shift_1_products'],row['shift_2_products']
        unq_products = unique_data_dict[user]
        for prod in unq_products:
            if prod in s2_products:
                if prod in s1_products:
                    if prod in products:
                        prob = stat_prob_1111.get(prod,np.nan)
                        cnt = stat_cnt_1111.get(prod,np.nan)
                    else:
                        prob = stat_prob_1101.get(prod,np.nan)
                        cnt = stat_cnt_1101.get(prod,np.nan)
                else:
                    if prod in s1_products:
                        prob = stat_prob_1011.get(prod,np.nan)
                        cnt = stat_cnt_1011.get(prod,np.nan)
                    else:
                        prob = stat_prob_1001.get(prod,np.nan)
                        cnt = stat_cnt_1001.get(prod,np.nan)
            else:
                if prod in s1_products:
                    if prod in s2_products:
                        prob = stat_prob_0111.get(prod,np.nan)
                        cnt = stat_cnt_0111.get(prod,np.nan)
                    else:
                        prob = stat_prob_0101.get(prod,np.nan)
                        cnt = stat_cnt_0101.get(prod,np.nan)
                else:
                    if prod in s2_products:
                        prob = stat_prob_0011.get(prod,np.nan)
                        cnt = stat_cnt_0011.get(prod,np.nan)
                    else:
                        prob = stat_prob_0001.get(prod,np.nan)
                        cnt = stat_cnt_0001.get(prod,np.nan)
            probs[(user,prod)] = [prob,cnt]
    probs_xxxx = pd.DataFrame.from_dict(probs,orient='index',columns=['prob_xxxx','cnt_xxxx']).reset_index()
    probs_xxxx['user_id'],probs_xxxx['product_id'] = zip(*probs_xxxx['index'])
    del probs_xxxx['index']
    
    suffix = 'test' if args.mode==0 else 'train'
    probs = probs_xx.merge(probs_xxx,how='left',on=['user_id','product_id']).merge(probs_xxxx,how='left',on=['user_id','product_id'])
    probs.to_csv(f'metadata/probs_xxx_{suffix}.csv',index=False)

#%%


