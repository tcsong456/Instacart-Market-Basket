# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:05:23 2024

@author: congx
"""
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
from sklearn.decomposition import NMF
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def adjacent_occurrence(data,items):
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
    rows = tqdm(order_products.iterrows(),total=order_products.shape[0],desc='collecting co-occurrence for adjacent baskets')
    for _,row in rows:
        products,next_products = row['product_id'],row['shift_products']
        if np.isnan(next_products).any():
            continue
        for next_prod in next_products:
            for prod in products:
                stats_cnt[(next_prod,prod)] += 1
    
    stats_cnt = pd.DataFrame.from_dict(stats_cnt,orient='index').reset_index().rename(columns={'index':'products',0:'cnt'})
    stats_cnt['target_prod'],stats_cnt['prod'] = zip(*stats_cnt['products'])
    del stats_cnt['products']
    stats_cnt = np.array(stats_cnt).astype(np.int32)
    interaction_matrix = np.zeros([items,items],dtype=np.int32)
    for c,i1,i2 in stats_cnt:
        i1 -= 1;i2 -= 1
        interaction_matrix[i1,i2] = c
    interaction_matrix = sparse.csr_matrix(interaction_matrix)

    return interaction_matrix

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    data = data[data['eval_set']!='test']
    items = data['product_id'].max()
    interaction_matrix = adjacent_occurrence(data,items)
    nmf = NMF(n_components=24,max_iter=1000)
    item_emb = nmf.fit_transform(interaction_matrix)
    feat_emb = nmf.components_
    item_ids = np.array([i for i in range(1,items+1)]).reshape(-1,1)
    item_emb = np.concatenate([item_ids,item_emb],axis=1).astype(np.float32)
    np.save('metadata/nmf_item_emb.npy',item_emb)
    
#%%
