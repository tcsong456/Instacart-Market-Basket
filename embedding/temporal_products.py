# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:42:30 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nn_model.product_lstm import ProductTemporalNet
from utils.utils import pickle_save_load,TMP_PATH
from itertools import chain
from embedding.trainer import Trainer

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()

def product_data_maker(data,max_len,prod_aisle_dict,prod_dept_dict,mode='train'):
    temp_dict = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{mode}.pkl'),mode='load')
        
    base_info = []
    data_dict = {}
    with tqdm(total=data.shape[0],desc='building product data for dataloader',dynamic_ncols=True,
              leave=False) as pbar:
        for _,row in data.iterrows():
            user,products = row['user_id'],row['product_id']
            reorders = row['reordered']
            days_interval = row['days_since_prior_order']
            if row['eval_set'] == 'test' and mode != 'test':
                products = products[:-1]
                reorders = reorders[:-1]
                days_interval = days_interval[:-1]
            days_interval = np.roll(days_interval,-1)[:-1]
            
            products,next_products = products[:-1],products[-1]
            next_products = next_products.split('_')
            orders = [prod.split('_') for prod in products]
            reorders_,next_reorders = reorders[:-1],reorders[-1]
            
            reorders_ = [list(map(int,reorder.split('_'))) for reorder in reorders_]
            all_products = list(set(chain.from_iterable(orders)))
            
            for prod in all_products:
                label = -1 if mode == 'test' else prod in next_products
                aisle = prod_aisle_dict[int(prod)]
                dept = prod_dept_dict[int(prod)]
                base_info.append([user,int(prod),aisle,dept])
                
                reorder_cnt = 0
                in_order_ord = []
                index_ord = []
                index_ratio_ord = []
                reorder_prod_ord = []
                reorder_prod_ratio_ord = []
                order_size_ord = []
                reorder_ord = []
                reorder_ratio_ord = []
                for idx,(order,reorder) in enumerate(zip(orders,reorders_)):
                    in_order = int(prod in order)
                    index_in_order = order.index(prod) + 1 if in_order else 0
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
                
                next_order_label = np.roll(in_order_ord,-1)
                next_order_label[-1] = label
                
                prod_info = np.stack([in_order_ord,np.array(order_size_ord)/145,np.array(index_ord)/145,index_ratio_ord,
                                      np.array(reorder_prod_ord)/100,reorder_prod_ratio_ord,np.array(reorder_ord)/130,reorder_ratio_ord,
                                      days_interval/30,next_order_label],axis=1).astype(np.float16)
                length = prod_info.shape[0]
                feature_dim = prod_info.shape[-1] - 1
                missing_seq = max_len - length
                if missing_seq > 0:
                    missing_data = np.zeros([missing_seq,prod_info.shape[1]],dtype=np.float16)
                    prod_info = np.concatenate([prod_info,missing_data])
                data_dict[(user,int(prod))] = (prod_info,length)
            
            base_info.append([user,0,0,0])
            reorder_cnt = 0
            in_order_ord = []
            index_ord = []
            index_ratio_ord = []
            reorder_prod_ord = []
            reorder_prod_ratio_ord = []
            order_size_ord = []
            reorder_ord = []
            reorder_ratio_ord = []
            next_reorders = list(map(int,next_reorders.split('_')))
            for idx,(order,reorder) in enumerate(zip(orders,reorders_)):
                in_order = int(max(reorder) == 0)
                index_in_order = 0
                order_size = len(order)
                index_order_ratio = 0
                reorder_cnt += in_order
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
            
            next_order_label = np.roll(in_order_ord,-1).reshape(-1,1)
            next_order_label[-1] = int(max(next_reorders)==0)
            prod_info = np.stack([in_order_ord,np.array(order_size_ord)/145,np.array(index_ord)/145,index_ratio_ord,
                                  np.array(reorder_prod_ord)/100,reorder_prod_ratio_ord,np.array(reorder_ord)/130,reorder_ratio_ord,
                                  days_interval/30]).transpose()
            prod_info = np.concatenate([prod_info,next_order_label],axis=1).astype(np.float16)
            length = prod_info.shape[0]
            feature_dim = prod_info.shape[-1] - 1
            missing_seq = max_len - length
            if missing_seq > 0:
                missing_data = np.zeros([missing_seq,prod_info.shape[1]],dtype=np.float16)
                prod_info = np.concatenate([prod_info,missing_data])
            data_dict[(user,0)] = (prod_info,length)
            
            pbar.update(1)
    user_prod = np.stack(base_info)
    
    return user_prod,data_dict,temp_dict,feature_dim

def product_dataloader(inp,
                       data_dict,
                       temp_dict,
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
    
    for batch in batch_gen():
        split_outputs = np.split(batch,batch.shape[1],axis=1)
        users,prods,aisles,depts = list(map(np.squeeze,split_outputs))
        dows,hours,tzs,days = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,length = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(length)
        temporals = [dows,hours,tzs,days]
        yield full_batch,batch_lengths,temporals,users-1,prods,aisles,depts

class ProductTrainer(Trainer):
    def __init__(self,
                  data,
                  prod_data,
                  output_dim,
                  learning_rate=0.01,
                  lagging=1,
                  optim_option='adam',
                  batch_size=32,
                  warm_start=False,
                  early_stopping=3,
                  epochs=100,
                  eval_epoch=1):
        super().__init__(data=data,
                         prod_data=prod_data,
                         output_dim=output_dim,
                         learning_rate=learning_rate,
                         lagging=lagging,
                         optim_option=optim_option,
                         batch_size=batch_size,
                         warm_start=warm_start,
                         early_stopping=early_stopping,
                         epochs=epochs,
                         eval_epoch=eval_epoch)
        self.dataloader = product_dataloader
        self.model = ProductTemporalNet
        self.model_name = self.model.__name__
        self.data_maker = product_data_maker
        self.emb_list = ['user_id','product_id','aisle_id','department_id']
        self.max_index_info = [data[col].max()+1 for col in self.emb_list] + [self.max_len]
        self.prod_aisle_dict = self.prod_data.set_index('product_id')['aisle_id'].to_dict()
        self.prod_dept_dict = self.prod_data.set_index('product_id')['department_id'].to_dict()
        self.data_maker_dicts = [self.prod_aisle_dict,self.prod_dept_dict]
        self.attr = 'product'
    
    def build_data_dl(self,mode):
        agg_data = self.build_agg_data('product_id')
        data_group = self.data_maker(agg_data,self.max_len,*self.data_maker_dicts,mode=mode)
        user_prod,data_dict,temp_dict,prod_dim = data_group
        for key,value in temp_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temp_dict[key] = value
        self.input_dim = self.temp_dim + prod_dim + len(self.emb_list) * 50
        self.agg_data = agg_data
        return [user_prod,data_dict,temp_dict]


if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')
        
    product_trainer = ProductTrainer(data,
                                    products,
                                    output_dim=50,
                                    eval_epoch=1,
                                    epochs=10,
                                    learning_rate=0.002,
                                    lagging=1,
                                    batch_size=512,
                                    early_stopping=2,
                                    warm_start=False,
                                    optim_option='adam')
    product_trainer.train(use_amp=False)
    product_trainer.predict(save_name='user_product_pred')



#%%

