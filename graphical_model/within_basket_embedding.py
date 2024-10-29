# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:58:07 2024

@author: congx
"""
import dgl
import torch
import itertools
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils import load_data
from torch.utils.data import Dataset,DataLoader
from torch import nn
from pandas.api.types import is_integer_dtype,is_float_dtype
from collections import defaultdict

class DNNSTP(nn.Module):
    def __init__(self,
                 n_items,
                 emb_dim):
        self.item_embedding = nn.Embedding(n_items,emb_dim)

class DnnstpSet(Dataset):
    def __init__(self,
                 data_path,
                 item_embedding):
        data_dict = load_data(data_path)
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
        
        from utils import optimize_dtypes
        for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
            if is_float_dtype(dtype):
                if pd.isnull(orders[col]).any():
                    orders[col].fillna(0,inplace=True)
                orders[col] = orders[col].astype(np.int64)
        orders = optimize_dtypes(orders)
        user_order_products = orders.groupby(['user_id','order_id','order_number'])['product_id'].apply(list)
        user_order_products = user_order_products.reset_index().sort_values(['user_id','order_number'])
        user_products = user_order_products.groupby(['user_id'])['product_id'].apply(list)
        self.user_products = user_products.to_dict()
        self.item_embedding = item_embedding
    
    def __getitem__(self,index):
        user_baskets = self.user_products[index+1]
        
        unique_items = torch.unique(torch.tensor(list(itertools.chain.from_iterable(user_baskets[:-1]))))
        products_embedding = self.item_embedding(unique_items)
        nodes = torch.tensor(list(range(products_embedding.shape[0])))
        src = np.stack([nodes for _ in range(nodes.shape[0])],axis=1).flatten().tolist()
        dst = np.stack([nodes for _ in range(nodes.shape[0])],axis=0).flatten().tolist()
        
        g = dgl.graph((src,dst),num_nodes=nodes.shape[0])
        edge_weight_dict = defaultdict(float)
        for basket in user_baskets:
            for i in range(len(basket)):
                itemi = basket[i]
                for j in range(i+1,len(basket)):
                    itemj = basket[j]
                    edge_weight_dict[(itemi,itemj)] += 1.0
                    edge_weight_dict[(itemj,itemi)] += 1.0

        for item in unique_items.tolist():
            if edge_weight_dict[(item,item)] == 0.0:
                edge_weight_dict[(item,item)] = 1.0
        max_weight = max(edge_weight_dict.values())
        for key,value in edge_weight_dict.items():
            edge_weight_dict[key] = value / max_weight

        edge_weights = []
        for basket in user_baskets[:-1]:
            edge_weight = []
            for item_1 in unique_items.tolist():
                for item_2 in unique_items.tolist():
                    if (item_1 in basket and item_2 in basket) or (item_1 == item_2):
                        edge_weight.append(edge_weight_dict[(item_1,item_2)])
                    else:
                        edge_weight.append(0.0)
            edge_weights.append(torch.Tensor(edge_weight))
        edge_weights = torch.stack(edge_weights)
        
        return g,products_embedding,edge_weight,unique_items,user_baskets


#%%
# import itertools
# import torch
# data_dict = load_data('data/')
# orders = data_dict['orders']
# orders_prior,orders_train = data_dict['order_products__prior'],data_dict['order_products__train']
# products = data_dict['products']
# aisles,departments = data_dict['aisles'],data_dict['departments']

# order_products = pd.concat([orders_prior,orders_train])
# del orders_prior,orders_train 
# orders = orders.merge(order_products,how='left',on='order_id').merge(products,\
#           how='left',on='product_id').merge(aisles,how='left',on='aisle_id').merge(departments,how='left',on='department_id')
# orders = orders[orders['eval_set']!='test']
# del order_products

# from utils import optimize_dtypes
# for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
#     if is_float_dtype(dtype):
#         if pd.isnull(orders[col]).any():
#             orders[col].fillna(0,inplace=True)
#         orders[col] = orders[col].astype(np.int64)
# orders = optimize_dtypes(orders)
# user_order_products = orders.groupby(['user_id','order_id','order_number'])['product_id'].apply(list)
# user_order_products = user_order_products.reset_index().sort_values(['user_id','order_number'])
# user_products = user_order_products.groupby(['user_id'])['product_id'].apply(list)

# user_products = user_products.to_dict()
# baskets = user_products[123]
# unique_items = torch.unique(torch.tensor(list(itertools.chain.from_iterable(baskets[:-1]))))
# item_emb = nn.Embedding((50000), 100)
# products_embedding = item_emb(unique_items)
# nodes = torch.tensor(list(range(products_embedding.shape[0])))
# src = np.stack([nodes for _ in range(nodes.shape[0])],axis=1).flatten().tolist()
# dst = np.stack([nodes for _ in range(nodes.shape[0])],axis=0).flatten().tolist()



#%%
model = DNNSTP(49688,100)

#%%
z = np.array(edge_weights)










