# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:39:29 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nn_model.reorder_lstm import ReorderLSTM
from embedding.trainer import Trainer
from utils.loss import NextBasketMSELoss,SeqMSELoss
from utils.utils import logger,pickle_save_load,TMP_PATH

convert_str_int = lambda x:list(map(int,x))
convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()
def reorder_data_maker(data,max_len,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_order','order_data_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    temp_dict = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{suffix}'),mode='load')
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'order_data_dict_{suffix}'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        feature_dim = data_dict[rand_key][0].shape[-1] - 1
        user_product = pickle_save_load(os.path.join(TMP_PATH,f'user_order_{suffix}'),mode='load')
        return user_product,data_dict,temp_dict,feature_dim
        
    base_info = []
    data_dict = {}
    with tqdm(total=data.shape[0],desc='building reorder data for dataloader',dynamic_ncols=True,
              leave=False) as pbar:
        for _,row in data.iterrows():
            user_id = row['user_id']
            days_interval = row['days_since_prior_order']
            base_info.append(user_id)
            reorders,aisles = row['reordered'],row['aisle_id']
            if row['eval_set'] == 'test' and mode != 'test':
                reorders = reorders[:-1]
                aisles = aisles[:-1]
                days_interval = days_interval[:-1]
            
            days_interval = np.roll(days_interval,-1)[:-1]
            reorders,next_reorders = reorders[:-1],reorders[-1]
            next_reorders = convert_str_int(next_reorders.split('_'))
            label = sum(next_reorders) if mode!='test' else 0
            reorders = [convert_str_int(reorder.split('_')) for reorder in reorders]
            
            total_reorders = 0
            total_size = 0
            reorder_sum_ord = []
            order_size_ord = []
            reorder_trend_ratio_ord = []
            reorder_ratio_ord = []
            order_size_avg_ord = []
            reorder_size_ratio_ord = []
            for idx,(reorder,aisle) in enumerate(zip(reorders,aisles)):
                reorders = sum(reorder)
                order_size = len(reorder)
                total_size += order_size
                reorder_ratio = reorders / order_size
                total_reorders += reorders
                reorder_size_ratio = total_reorders / total_size
                reorder_trend_ratio = np.log1p(total_reorders / (idx + 1))
                order_size_avg = np.log1p(total_size / (idx + 1))
                
                reorder_sum_ord.append(np.log1p(reorders))
                order_size_ord.append(np.log1p(order_size))
                reorder_trend_ratio_ord.append(reorder_trend_ratio)
                reorder_ratio_ord.append(reorder_ratio)
                reorder_size_ratio_ord.append(reorder_size_ratio)
                order_size_avg_ord.append(order_size_avg)
            
            next_in_reorder = np.roll(reorder_sum_ord,-1)
            next_in_reorder[-1] = np.log1p(label)
            
            reorder_info = np.stack([reorder_sum_ord,np.array(order_size_ord)/145,
                                     reorder_ratio_ord,reorder_size_ratio_ord,reorder_trend_ratio_ord,order_size_avg_ord,
                                     days_interval/30,next_in_reorder],axis=1).astype(np.float16)
            reorder_info_dim = reorder_info.shape[1] - 1
            length = reorder_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,reorder_info_dim+1],dtype=np.float16)
            reorder_info = np.concatenate([reorder_info,paddings])
            
            data_dict[user_id] = (reorder_info,length)
            pbar.update(1)
    users = np.array(base_info).reshape(-1,1)
    
    save_data  = [users,data_dict]
    for path,file in zip(save_path,save_data):
        path = os.path.join(TMP_PATH,path)
        pickle_save_load(path,file,mode='save') 	
    
    return users,data_dict,temp_dict,reorder_info_dim

def reorder_dataloader(inp,
                     data_dict,
                     temp_dict,
                     batch_size=32,
                     shuffle=True,
                     drop_last=False):
    def batch_gen():
        total_length = inp.shape[0]
        indices = np.arange(total_length)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0,total_length,batch_size):
            ind = indices[i:i+batch_size]
            if len(ind) < batch_size and drop_last:
                break
            else:
                batch = inp[ind]
                yield batch
    
    for batch in batch_gen():
        users = batch.squeeze()
        dows,hours,tzs,days = zip(*list(map(temp_dict.get,users)))
        keys = np.split(batch,batch.shape[0],axis=0)
        keys = list(map(lambda x:x[0][0],keys))
        data,data_len = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(data_len)
        temporals = [dows,hours,tzs,days]
        yield full_batch,batch_lengths,temporals,users-1
        
class ReorderTrainer(Trainer):
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
        self.dataloader = reorder_dataloader
        self.data_maker = reorder_data_maker
        self.model = ReorderLSTM
        self.model_name = self.model.__class__.__name__
        self.attr = 'reorder'
        
        self.emb_list = ['user_id']
        self.max_index_info = [data[col].max() for col in self.emb_list] + [self.max_len]
        self.data_maker_dicts = []
        
        self.loss_fn_tr = SeqMSELoss(lagging=lagging)
        self.loss_fn_te = NextBasketMSELoss(lagging=lagging)
    
    def build_data_dl(self,mode):
        agg_data = self.build_agg_data('aisle_id')
        data_group = self.data_maker(agg_data,self.max_len,*self.data_maker_dicts,mode=mode)
        user_aisle,data_dict,temp_dict,aisle_dim = data_group
        for key,value in temp_dict.items():
            for i in range(len(value)):
                value[i] = torch.Tensor(value[i]).long().cuda()
            temp_dict[key] = value
        self.input_dim = self.temp_dim + aisle_dim + len(self.emb_list) * 50
        self.agg_data = agg_data
        return [user_aisle,data_dict,temp_dict]

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')

    trainer = ReorderTrainer(data,
                            products,
                            output_dim=50,
                            lagging=1,
                            learning_rate=0.002,
                            optim_option='adam',
                            batch_size=64,
                            warm_start=False,
                            early_stopping=2,
                            epochs=10,
                            eval_epoch=1)
    trainer.train(use_amp=False,ev='')
    trainer.predict(save_name='user_reorder_pred',ev='')

#%%