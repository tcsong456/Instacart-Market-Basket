# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:41:30 2024

@author: congx
"""
import warnings
warnings.filterwarnings('ignore')
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter,defaultdict
from itertools import chain
from utils import optimize_dtypes,load_data
from pandas.api.types import is_float_dtype

class Sampler:
    def __init__(self,
                 path,
                 mode='train'):
        data_dict = load_data(path)
        orders = data_dict['orders']
        orders_prior,orders_train = data_dict['order_products__prior'],data_dict['order_products__train']
        products = data_dict['products']
        aisles,departments = data_dict['aisles'],data_dict['departments']
        
        order_products = pd.concat([orders_prior,orders_train])
        del orders_prior,orders_train 
        orders = orders.merge(order_products,how='left',on='order_id').merge(products,\
                  how='left',on='product_id').merge(aisles,how='left',on='aisle_id').merge(departments,how='left',on='department_id')
        orders = orders[orders['eval_set']!='test']
        del order_products
        for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
            if is_float_dtype(dtype):
                if pd.isnull(orders[col]).any():
                    orders[col].fillna(0,inplace=True)
                orders[col] = orders[col].astype(np.int64)
        orders = optimize_dtypes(orders)
        
        order_products = orders.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
        order_products = order_products.sort_values(['user_id','order_number'])
        user_products = order_products.groupby('user_id')['product_id'].apply(list).reset_index()
        order_products = user_products.copy()
        order_products['product_id'] = order_products['product_id'].apply(lambda x:self._return_list(x,mode))
        order_products = order_products.explode('product_id')
        user_products['product_id'] = user_products['product_id'].apply(lambda x:list(set(chain.from_iterable(self._return_list(x,mode)))))
        
        item_freq = Counter(orders['product_id'])
        item_freq = pd.DataFrame.from_dict(item_freq,orient='index',columns=['count'])
        item_freq['count'] = (item_freq['count']**0.75).apply(np.floor)
        item_freq['count'] = item_freq['count'] / item_freq['count'].sum()
        item_freq_dict = item_freq['count'].to_dict()
        missing_products = set(products['product_id'].tolist()) - set(item_freq_dict.keys())
        for m in missing_products:
            item_freq_dict[m] = 0
        
        item_coexist_dict = self._item_coexist_dict(order_products)
        item_item = pd.DataFrame(list(item_coexist_dict.items()),columns=['product_pair','value'])
        item_item = pd.DataFrame(item_item['product_pair'].tolist())
        user_item = user_products.explode('product_id')
        self.user_item_nodes = np.array(user_item).astype(np.int32)
        self.item_item_nodes = np.array(item_item).astype(np.int32)
                                                          
        
        self.orders = orders
        self.products = products
        self.aisles = aisles
        self.departments = departments
        self.user_products = user_products
        self.order_products = order_products
        self.item_freq_dict = item_freq_dict
        self.item_coexist_dict = item_coexist_dict
        
    def _return_list(self,x,mode):
        if mode == 'test' or 'emb':
            return x
        else:
            if mode == 'train':
                return x[:-2]
            else:
                return x[:-1]
    
    def _item_coexist_dict(self,order_products):
        rows = tqdm(order_products.iterrows(),total=order_products.shape[0],desc='buiding item co-existent dict')
        coexist_dict = defaultdict(int)
        for _,row in rows:
            unique_items = row['product_id']
            for item1 in unique_items:
                for item2 in unique_items:
                    if item1 != item2 and (item1,item2):
                        coexist_dict[(item1,item2)] += 1
        return coexist_dict
    
    def user_neg_samples(self,num_neg_users=5):
        items = list(self.item_freq_dict.keys())
        sorted_items = np.argsort(items)
        items_prob = list(self.item_freq_dict.values())
        items_prob = np.array(items_prob)
        items_prob = items_prob[sorted_items]
        
        neg_samples = torch.zeros(self.user_products.shape[0],num_neg_users,dtype=torch.int32)
        rows = tqdm(self.user_products.iterrows(),total=self.user_products.shape[0],desc='sampling negative samples for the user')
        for _,row in rows:
            probs = items_prob.copy()
            user,pos_item = row['user_id'],np.array(row['product_id']) - 1
            probs[pos_item] = 0.0
            valid_neg_users = probs.nonzero()[0].shape[0]
            num_neg_users = min(valid_neg_users,num_neg_users)
            probs = torch.from_numpy(probs)
            neg_sample = torch.multinomial(probs,num_neg_users,replacement=False)
            neg_samples[user-1] = neg_sample + 1
        return np.array(neg_samples)
    
    def item_neg_samples(self,num_neg_aisles=5,num_neg_items=100):
        products = self.products
        item_item_df = pd.DataFrame(self.item_item_nodes,columns=['prod_src','product_id'])
        pos_items = item_item_df.groupby('prod_src')['product_id'].apply(list)
        item_products = item_item_df.merge(products,how='left',on=['product_id'])
        item_coexist_df = pd.DataFrame.from_dict(self.item_coexist_dict,orient='index',columns=['cnt']).reset_index().rename(columns={'index':'prod'})
        item_coexist_df['prod_src'],item_coexist_df['product_id'] = zip(*item_coexist_df['prod'])
        del item_coexist_df['prod']
        item_products = item_products.merge(item_coexist_df,how='inner',on=['prod_src','product_id'])

        t = item_products.groupby(['prod_src','aisle_id'])['cnt'].sum().reset_index()
        t1 = t.groupby('prod_src')['cnt'].apply(list)
        t2 = t.groupby('prod_src')['aisle_id'].apply(list)
        t = pd.concat([t1,t2],axis=1)
        t = t.reset_index()
        del t1,t2,item_products
        gc.collect()
        
        pos_item_dict = pos_items.to_dict()
        rows = tqdm(t.iterrows(),total=t.shape[0],desc='building nagetive item samples')
        neg_samples = np.zeros([t['prod_src'].max(),num_neg_items],dtype=np.int32)
        products['prob'] = products['product_id'].map(self.item_freq_dict)
        for _,row in rows:
            aisles,cnt,prod = np.array(row['aisle_id']),row['cnt'],row['prod_src']
            norm_cnt = np.array(cnt) / sum(cnt)
            norm_cnt = torch.from_numpy(norm_cnt)
            neg_aisle = min(len(norm_cnt),num_neg_aisles)
            aisle = torch.multinomial(norm_cnt,neg_aisle,replacement=False)
            aisles = aisles[aisle]
            aisles = aisles if hasattr(aisles,'__iter__') else [aisles]
            items = products[products['aisle_id'].isin(aisles)]
            selected_items = np.array(items['product_id'].tolist())
            pos_item = np.array(pos_item_dict[prod])
            common_items = np.intersect1d(selected_items,pos_item).tolist()
            dist = items['prob'].tolist()
            while common_items:
                remove_item = common_items.pop()
                index = np.where(selected_items==remove_item)[0][0]
                dist[index] = 0
            num_neg = min(items.shape[0],num_neg_items)
            sampled_index = torch.multinomial(torch.Tensor(dist),num_neg,replacement=False)
            sampled_items = selected_items[sampled_index]
            neg_samples[prod-1,:num_neg] = sampled_items
            
            if num_neg_items > items.shape[0]:
                filled_items = num_neg_items - items.shape[0]
                non_items = products[~products['aisle_id'].isin(aisles)]
                nitems = non_items['product_id']
                pure_items = list(set(nitems) - set(pos_item))
                non_items = non_items[non_items['product_id'].isin(pure_items)]
                pure_items,pure_probs = np.array(non_items['product_id']),np.array(non_items['prob'])
                indices = torch.multinomial(torch.from_numpy(pure_probs),filled_items,replecement=False)
                extra_items = pure_items[indices]
                neg_samples[prod-1,num_neg:] = extra_items
            
        return neg_samples

data_sampler = Sampler('data',mode='emb')

#%%
