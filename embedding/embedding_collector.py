# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:52:16 2024

@author: congx
"""
import gc
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from utils.utils import pickle_save_load

def collector(emb_loss='bpr',mode='evaluate',top=3):
    agg_data = pd.read_pickle('data/tmp/user_product_info.csv')
    # if mode == 'evaluate':
    #     agg_data_tr,agg_data_te = agg_data[agg_data['eval_set']=='train'],agg_data[agg_data['eval_set']=='test']
    #     agg_data_tr = agg_data_tr[agg_data_tr['product_id'].map(lambda x:len(x)-2>=2)]
    #     agg_data_te = agg_data_te[agg_data_te['product_id'].map(lambda x:len(x)-3>=2)]
    #     agg_data_ = pd.concat([agg_data_tr,agg_data_te])
    #     del agg_data_tr,agg_data_te
    #     gc.collect()
    # else:
    #     agg_data_ = agg_data
    
    if emb_loss == 'bpr':
        indicator = 0
        item_embedding = 0
        for file in os.listdir('checkpoint'):
            if 'BPR' in file:
                indicator += 1
                state_dict = torch.load(f'checkpoint/{file}')
                emb = state_dict['model_state_dict']['item_embedding.weight'].detach().cpu().numpy().astype(np.float16)
                item_embedding += emb
        item_embedding /= indicator
    
    data_eval = []
    stats_cnt = pickle_save_load('data/tmp/product_stats_cnt.pkl',mode='load')
    rows = tqdm(agg_data.iterrows(),total=agg_data.shape[0],desc='collecting training item_embedding')
    for _,row in rows:
        user,products = row['user_id'],row['product_id']
        if mode == 'evaluate' and row['eval_set'] == 'test':
            products = products[:-1]
        
        products = products[:-1]
        orders = [list(map(int,prod.split('_'))) for prod in products]
        all_products = list(set(chain.from_iterable(orders)))
        eval_order = orders[-2] if mode=='evaluate' else orders[-1]
        for prod in all_products:
            keys = [(prod,r) for r in eval_order]
            cnts = list(map(lambda key:stats_cnt.get(key,0),keys))
            sort_index = np.argsort(cnts)[-1]
            top_items = np.array(eval_order)[sort_index] - 1
            item_embs = item_embedding[top_items]
            # missing_len = top - len(sort_index)
            # for _ in range(missing_len):
            #     filler = np.zeros([1,item_embs.shape[1]],dtype=np.float16)
            #     item_embs = np.concatenate([item_embs,filler])
            # item_embs = item_embs.reshape(-1)
            drow = np.concatenate([np.array([user,prod]),item_embs]).astype(np.float32)
            data_eval.append(drow)
        drow = np.concatenate([np.array([user,0]),np.zeros_like(item_embs)]).astype(np.float32)
        data_eval.append(drow)
    
    data_pred = []
    agg_data_te = agg_data[agg_data['eval_set']=='test']
    rows = tqdm(agg_data_te.iterrows(),total=agg_data_te.shape[0],desc='collecting testing item_embedding')
    for _,row in rows:
        user,products = row['user_id'],row['product_id']
        if mode == 'evaluate':
            products = products[:-1]
        
        orders = [list(map(int,prod.split('_'))) for prod in products]
        all_products = list(set(chain.from_iterable(orders)))
        
        pred_order = orders[-2]
        for prod in all_products:
            keys = [(prod,r) for r in pred_order]
            cnts = list(map(lambda key:stats_cnt.get(key,0),keys))
            sort_index = np.argsort(cnts)[-1]
            top_items = np.array(pred_order)[sort_index] - 1
            item_embs = item_embedding[top_items]
            # missing_len = top - len(sort_index)
            # for _ in range(missing_len):
            #     filler = np.zeros([1,item_embs.shape[1]],dtype=np.float16)
            #     item_embs = np.concatenate([item_embs,filler])
            # item_embs = item_embs.reshape(-1)
            drow = np.concatenate([np.array([user,prod]),item_embs]).astype(np.float32)
            data_pred.append(drow)
        drow = np.concatenate([np.array([user,0]),np.zeros_like(item_embs)]).astype(np.float32)
        data_pred.append(drow)
    
    for dn,suffix in zip([data_eval,data_pred],['eval','pred']):
        np.save(f'metadata/evaluation/user_prod_emb_{suffix}.npy',dn)

def emb_dataloader(
                   user_prod_emb,
                   user_prod,
                   user_aisle,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   mode='train'):
    def batch_gen():
        total_len = user_prod.shape[0]
        index = np.arange(total_len)
        if shuffle:
            np.random.shuffle(index)
        for i in range(0,total_len,batch_size):
            idx = index[i:i+batch_size]
            if len(idx) < batch_size and drop_last:
                break
            else:
                yield idx
    if mode == 'train':
        user_prod,label = user_prod[:,:-1],user_prod[:,-1]
    else:
        label = None
        
    # for batch in batch_gen():
    #     batch_emb = user_prod_emb[idx]
    #     batch_prod = user_prod[idx]
        
    # for batch in batch_gen():
    #     keys = np.split(keys,keys.shape[0],axis=0)
    #     keys = list(map(tuple,(map(np.squeeze,keys))))
        

# if __name__ == '__main__':
#     collector(emb_loss='bpr',mode='evaluate',top=3)
    # products = pd.read_csv('data/products.csv')
    # convert_to_df = lambda x,t:pd.DataFrame(x,columns=['user_id',t]+[f'f{i}' for i in range(x.shape[1]-2)])
    # user_prod_eval = np.load('metadata/evaluation/user_product_eval.npy')
    # user_prod_pred = convert_to_df(np.load('metadata/evaluation/user_product_pred.npy'),'product_id')
    # user_prod_emb_eval = np.load('metadata/evaluation/user_prod_emb_eval.npy')
    # z = user_prod_emb_eval[:10000]
    # user_prod_emb_pred = convert_to_df(np.load('metadata/evaluation/user_prod_emb_pred.npy'),'product_id')
    # user_aisle_eval = convert_to_df(np.load('metadata/evaluation/user_aisle_eval.npy'),'aisle_id')
    # user_aisle_pred = convert_to_df(np.load('metadata/evaluation/user_aisle_pred.npy'),'aisle_id')
    # user_prod_eval,label = user_prod_eval[:,:-1],user_prod_eval[:,-1]
    # user_prod_eval = convert_to_df(user_prod_eval,'product_id')
    
    # prod_aisle_dict = products.set_index('product_id')['aisle_id'].to_dict()
    # user_prod_eval['aisle_id'] = user_prod_eval['product_id'].map(prod_aisle_dict)
    # user_prod_pred['asile_id'] = user_prod_pred['product_id'].map(prod_aisle_dict)
    
    # x_eval = user_prod_eval.merge(user_aisle_eval,how='left',on=['user_id','aisle_id']).merge(user_prod_emb_eval,
    #                                                             how='left',on=['user_id','product_id'])
    # del x['aisle_id']
            
#%%
indicator = 0
item_embedding = 0
for file in os.listdir('checkpoint'):
    if 'BPR' in file:
        indicator += 1
        state_dict = torch.load(f'checkpoint/{file}')
        emb = state_dict['model_state_dict']['item_embedding.weight'].detach().cpu().numpy().astype(np.float16)
        item_embedding += emb
item_embedding /= indicator
#%%
# agg_data = pd.read_pickle('data/tmp/user_product_info.csv')
# stats_cnt = pickle_save_load('data/tmp/product_stats_cnt.pkl',mode='load')
rows = tqdm(agg_data.iterrows(),total=agg_data.shape[0],desc='collecting training item_embedding')
for idx,row in rows:
    user,products = row['user_id'],row['product_id']
    if row['eval_set'] == 'test':
        products = products[:-1]
    products = products[:-1]
    orders = [list(map(int,prod.split('_'))) for prod in products]
    all_products = list(set(chain.from_iterable(orders)))
    eval_order = orders[-2]
    for ind,prod in enumerate(all_products):
        keys = [(prod,r) for r in eval_order]
        cnts = list(map(lambda key:stats_cnt.get(key,0),keys))
        sort_index = np.argsort(cnts)[-1]
        if ind == 2:
            break
    # if idx == 1:
    break

