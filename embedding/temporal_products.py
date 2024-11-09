# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:42:30 2024

@author: congx
"""
import torch
import warnings
import numpy as np
import pandas as pd
from utils import Timer,logger
from itertools import chain
from nn_model.lstm import ProdLSTM
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def data_processing(data):
    data = data.sort_values(['user_id','order_number','add_to_cart_order'])
    grouped_data = []
    
    for col in ['product_id','reordered','aisle_id','department_id']:
        x = data.groupby(['user_id','order_id','order_number'])[col].apply(list).map(lambda x:'_'.join(map(str,x))).reset_index()
        x = x.sort_values(['user_id','order_number']).groupby(['user_id'])[col].apply(list)
        grouped_data.append(x)
    r = pd.concat(grouped_data,axis=1)
    
    grouped_attr = []
    for col in ['order_hour_of_day','order_dow','days_since_prior_order','time_zone']:
        x = data.groupby(['user_id','order_id','order_number']).apply(lambda x:x[col].iloc[0])
        x.name = col
        x = x.reset_index().sort_values(['user_id','order_number']).groupby('user_id')[col].apply(list).map(lambda x:'_'.join(map(str,x)))
        grouped_attr.append(x)
    x = pd.concat(grouped_attr,axis=1)
    r = r.merge(x,how='left',on='user_id').reset_index()
    return r

convert_format = lambda x:list(map(int,x.split('_')))
def oh_encoding(target,num_cates):
    assert len(target.shape) == 1,'target must be a 1-D array'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        oh = OneHotEncoder(categories=[np.arange(num_cates)],sparse=False)
        target = target.reshape(-1,1)
        ohc = oh.fit_transform(target)
    return ohc

def data_for_training(data,max_len):
    data_dict = {}
    user_ids,product_ids = [],[]
    for _,row in data.iterrows():
        user,products = row['user_id'],row['product_id']
        reorders = row['reordered']
        aisles,departments = row['aisle_id'],row['department_id']
        orders = [product.split('_') for product in products]
        products,next_products = products[:-1],products[-1]
        reorders = reorders[:-1]
        reorders = [list(map(int,reorder.split('_'))) for reorder in reorders]
        all_products = [list(set(chain.from_iterable(product.split('_') for product in products)))][0]
        
        labels = []
        for product in all_products:
            user_ids.append(user)
            product_ids.append(int(product))
            label = product in next_products
            labels.append(label)
            
            reorder_cnt = 0
            in_order_ord = []
            index_ord = []
            index_ratio_ord = []
            reorder_prod_ord = []
            reorder_prod_ratio_ord = []
            order_size_ord = []
            reorder_ord = []
            reorder_ratio_ord = []
            for idx,(order,reorder,aisle,department) in enumerate(zip(orders,reorders,aisles,departments)):
                in_order = int(product in order)
                index_in_order = order.index(product) + 1 if in_order else 0
                order_size = len(order)
                index_order_ratio = index_in_order / order_size
                reorder_cnt += int(in_order)
                reorder_ratio_cum = reorder_cnt / (idx+1)
                reorder_size = sum(reorder)
                reorder_ratio = reorder_size / order_size
                
                in_order_ord.append(in_order)
                order_size_ord.append(order_size)
                index_ord.append(index_in_order)
                index_ratio_ord.append(index_order_ratio)
                reorder_prod_ord.append(reorder_cnt)
                reorder_prod_ratio_ord.append(reorder_ratio_cum)
                reorder_ord.append(reorder_size)
                reorder_ratio_ord.append(reorder_ratio)
            
            order_dow = oh_encoding(np.roll(convert_format(row['order_dow']),-1)[:-1],7)
            order_hour = oh_encoding(np.roll(convert_format(row['order_hour_of_day']),-1)[:-1],24)
            days_interval = np.roll(convert_format(row['days_since_prior_order']),-1)[:-1]
            tz = oh_encoding(np.roll(convert_format(row['time_zone']),-1)[:-1],28)
            prod_info = np.stack([in_order_ord,order_size_ord,index_ord,index_ratio_ord,
                                  reorder_prod_ord,reorder_prod_ratio_ord,reorder_ord,reorder_ratio_ord,
                                  days_interval]).transpose()
            prod_info = np.concatenate([prod_info,order_dow,order_hour,tz],axis=1).astype(np.float16)
            length = prod_info.shape[0]
            feature_dim = prod_info.shape[1]
            missing_seq = max_len - length
            if missing_seq > 0:
                missing_data = np.zeros([missing_seq,prod_info.shape[1]],dtype=np.float16)
                prod_info = np.concatenate([prod_info,missing_data])
            
            data_dict[(user,int(product))] = (prod_info,length)
    user_prod = np.stack([user_ids,product_ids],axis=1)
    return user_prod,data_dict

def product_dataloader(inp,
                       data_dict,
                       prod_data,
                       batch_size=32,
                       shuffle=True,
                       drop_last=True
                       ):
    def batch_gen():
        total_len = inp.shape[0]
        index = np.arange(total_len)
        if shuffle:
            np.random.shuffle(index)
        for i in range(0,total_len,batch_size):
            idx = index[i:i+batch_size]
            if len(idx) < batch_size and drop_last:
                break
            else:
                yield inp[idx]
    
    prod_aisle_dict = prod_data.set_index('product_id')['aisle_id'].to_dict()
    prod_dep_dict = prod_data.set_index('product_id')['department_id'].to_dict()
    for batch in batch_gen():
        full_batch = []
        batch_lengths = []
        users,prods,aisles,depts = [],[],[],[]
        for row in batch:
            user_id,product_id = row
            users.append(user_id)
            prods.append(product_id)
            aisles.append(prod_aisle_dict[product_id])
            depts.append(prod_dep_dict[product_id])
            key = (user_id,product_id)
            data,length = data_dict[key]
            full_batch.append(data)
            batch_lengths.append(length)
        full_batch = np.stack(full_batch)
        yield full_batch,batch_lengths,users,prods,aisles,depts

def trainer(model):
    

if __name__ == '__main__':
    import sys
    # try:
    #     data = pd.read_csv('data/orders_info.csv')
    #     z = data.iloc[:10000]
    # except FileNotFoundError as e:
    #     logger.warning(f"{e}! Please create the data using 'create_merged_data' file first")
    #     sys.exit(1)
    # products = pd.read_csv('data/products.csv')
        
    with Timer():
        # agg_data = data_processing(z)
        # pre_data = data_for_training(agg_data,max_len=100)
        
        # max_user = data['user_id'].max()
        # max_product = products['product_id'].max()
        # max_aisle = products['aisle_id'].max()
        # max_dept = products['department_id'].max()
        # max_index_info = [max_user,max_product,max_aisle,max_dept]
        # dl = product_dataloader(pre_data[0],pre_data[1],products)
        model = ProdLSTM(268,64,50,*max_index_info,num_layers=1,batch_first=True,dropout=0.0).to('cuda')
        for batch,lengths,*aux_info in dl:
            batch = torch.from_numpy(batch).cuda()
            input_dim = batch.shape[-1] + 4 * 50
            aux_info = [torch.Tensor(b).long().to('cuda') for b in aux_info]
            output = model(batch,lengths,*aux_info)
            break



#%%
pre_data[1]

