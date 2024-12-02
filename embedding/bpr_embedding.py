# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:39:09 2024

@author: congx
"""
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def pos_pair_cnt(data):
    data = data[data['eval_set']!='test']
    order_products = data.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    order_products = order_products.sort_values(['user_id','order_number'])
    del order_products['order_number']
    shift_order_products = order_products.groupby('user_id')['product_id'].shift(-1)
    shift_order_products.name = 'shift_products'
    order_products = pd.concat([order_products,shift_order_products],axis=1)
    del data
    gc.collect()
    
    stats_cnt = defaultdict(int)
    rows = tqdm(order_products.iterrows(),total=order_products.shape[0],desc='collecting count stats for positive pairs')
    for _,row in rows:
        products,next_products = row['product_id'],row['shift_products']
        if np.isnan(next_products).any():
            continue
        for next_prod in next_products:
            for prod in products:
                stats_cnt[(next_prod,prod)] += 1
        
    return order_products,stats_cnt

def pos_pair_stats(data,pos_cnt_dict,top=3):
    pos_pair = defaultdict(list)
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='collecting positive pairs stats')
    for _,row in rows:
        products,next_products = row['product_id'],row['shift_products']
        if np.isnan(next_products).any():
            continue
        for next_prod in next_products:
            keys = [(next_prod,prod) for prod in products]
            cnts = list(map(lambda k:pos_cnt_dict.get(k,0),keys))
            sort_index = np.argsort(cnts)[::-1]
            top_index = sort_index[:top] if len(cnts) > top else sort_index
            keys = np.array(keys)
            pos_pairs = keys[top_index].tolist()
            pos_items = list(map(lambda x:x[1],pos_pairs))
            pos_pair[next_prod] += pos_items
    return pos_pair

def neg_pair_stats(data,pos_pair_dict):
    data = data[data['eval_set']!='test']
    prods = set(data['product_id'])
    prod_prob = data['product_id'].value_counts().to_dict()
    for key,value in prod_prob.items():
        prod_prob[key] = np.array((value / data.shape[0])**0.75).astype(np.float32)
    
    neg_pair = defaultdict(list)
    neg_pair_prob = defaultdict(list)
    items = tqdm(pos_pair_dict.items(),total=len(pos_pair_dict.keys()))
    for key,value in items:
        neg_prods = prods - set(value)
        neg_pair[key] = list(neg_prods)
        neg_prob = np.array(list(map(lambda x:prod_prob.get(x,0),neg_prods)))
        neg_prob /= neg_prob.sum()
        neg_pair_prob[key] = neg_prob.tolist()
    return neg_pair,neg_pair_prob
                
def emb_data_maker(pos_pair,neg_pair,neg_pair_prob,num_neg=5):
    samples = []
    for key,value in tqdm(pos_pair.items(),total=len(pos_pair.keys())):
        neg_samples = np.random.choice(neg_pair[key],p=neg_pair_prob[key],replace=True,size=len(value)*num_neg).reshape(-1,num_neg)
        pos_samples = np.array(value).reshape(-1,1)
        target_samples = np.array([key] * neg_samples.shape[0]).reshape(-1,1)
        sample = np.concatenate([target_samples,pos_samples,neg_samples],axis=1)
        samples.append(sample)

    samples = np.concatenate(samples)
    return samples

def emb_dataloader(prod_samples,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False
                   ):
    def batch_gen():
        total_length = prod_samples.shape[0]
        indices = np.arange(total_length)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0,total_length,batch_size):
            ind = indices[i:i+batch_size]
            if len(ind) < batch_size and drop_last:
                break
            else:
                batch = prod_samples[ind]
                yield batch
    
    for batch in batch_gen():
        batch = torch.from_numpy(batch).long().cuda()
        yield batch
    
class EmbeddingTrainer:
    def __init__(self,
                 data,
                 batch_size):
        self.data = data
        self.batch_size = batch_size

if __name__ == '__main__':
    # agg_data = pd.read_pickle('data/tmp/user_product_info.csv')
    # data = pd.read_csv('data/orders_info.csv')
    # order_products,stats_cnt = pos_pair_cnt(data)
    # pos_pair = pos_pair_stats(order_products,stats_cnt,top=3)
    # neg_pair,neg_pair_probs = neg_pair_stats(data,pos_pair)
    # prods = emb_data_maker(pos_pair,neg_pair,neg_pair_probs,num_neg=5)
    # dl = emb_dataloader(prods,batch_size=256)
    


#%%
from itertools import product
from functools import partial
from utils.utils import Timer
import torch
# data = data[data['eval_set']!='test']
# z = data.iloc[:1000000]
# x = z.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
# x = x.sort_values(['user_id','order_number'])
# c = x.groupby('user_id')['product_id'].shift(-1)
# c.name = 'shift_products'
# products = pd.read_csv('data/products.csv')

# d = defaultdict(list)
with Timer(5):
    for batch in dl:
        break
#%%
x = np.random.choice(neg_pair[196],p=neg_pair_probs[196],replace=True,size=100000*5).reshape(-1,5)

#%%
batch.shape