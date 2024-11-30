# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:50:57 2024

@author: congx
"""
import gc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from utils.utils import pad,logger,TMP_PATH,pickle_save_load,optimize_dtypes
from nn_model.aisle_lstm import AisleLSTM
from embedding.trainer import Trainer
from more_itertools import unique_everseen
from embedding.base_stats_collector import BaseStatsCollector
from collections import defaultdict

class AisleStatsCollector(BaseStatsCollector):
    def __init__(self,
                 data):
        super().__init__(data=data)
        user_unique_aisles = data[data['eval_set']!='test'].groupby('user_id')['aisle_id'].apply(set).apply(list)
        self.unique_dict = user_unique_aisles.to_dict()
        self.stats = self.build_data()
    
    def build_data(self):
        path = os.path.join(TMP_PATH,'aisle_stats_data.pkl')
        try:
            stats = pd.read_pickle(path)
        except FileNotFoundError:
            stats = []
            order_aisle = self.data.groupby(['user_id','order_id','order_number','aisle_id'])['counter'].sum().reset_index()
            order_aisle = order_aisle.sort_values(['user_id','order_number'])
            order_aisle_ls = order_aisle.groupby(['user_id','order_id','order_number'])['aisle_id'].apply(list).reset_index()
            order_aisle_ls = order_aisle_ls.sort_values(['user_id','order_number'])
            del order_aisle_ls['order_number']
            stats.append(order_aisle_ls)
            
            order_aisle_shift = order_aisle_ls.groupby('user_id')['aisle_id'].shift(-1)
            order_aisle_shift.name = 'aisle_next'
            stats.append(order_aisle_shift)
            
            order_aisle_cnt = order_aisle.groupby(['user_id','order_id','order_number'])['counter'].apply(list).reset_index()
            order_aisle_cnt = order_aisle_cnt.sort_values(['user_id','order_number'])
            del order_aisle_cnt['order_number']
            stats.append(order_aisle_cnt[['counter']])
            
            for col in ['order_dow','days_since_prior_order']:
                s = self._shift_stats(col)
                stats.append(s)
            stats = pd.concat(stats,axis=1)
            stats = optimize_dtypes(stats)
            stats.to_pickle(path)
        return stats
    
    def _shift_stats(self,col):
        stat = self.data.groupby(['user_id','order_id','order_number'])[col].apply(lambda x:x.iloc[-1]).reset_index()
        stat = stat.sort_values(['user_id','order_number'])
        del stat['order_number']
        stat = stat.groupby(['user_id'])[col].shift(-1)
        stat.name = col
        return stat
    
    def true_adjacent_stat(self,valid_cnt=1,min_interval=0,max_interval=31,comparator='equal'):
        operators = {'equal':operator.eq,
                     'greater_equal':operator.ge}
        op = operators[comparator]
        true_stats = defaultdict(lambda:defaultdict(int))
        true_interval_stats = defaultdict(list)
        rows = tqdm(self.stats.iterrows(),total=self.stats.shape[0],desc='building true adjacent row stat')
        for _,row in rows:
            user,cur_aisles,next_aisles,cnts = row['user_id'],row['aisle_id'],row['aisle_next'],row['counter']
            if np.isnan(next_aisles).any() or next_aisles == [-1]:
                continue
            order_interval = row['days_since_prior_order']
            unique_aisles = self.unique_dict[user]
            for aisle in unique_aisles:
                if aisle in cur_aisles :
                    index = cur_aisles.index(aisle)
                    cnt = cnts[index]
                    if op(cnt,valid_cnt) and order_interval >= min_interval and order_interval < max_interval:
                        key = f'{str(valid_cnt)}_{str(min_interval)}_{str(max_interval)}'
                        true_stats[aisle][f'total_{key}'] += 1
                        if aisle in next_aisles:
                            true_stats[aisle][f'in_{key}'] += 1
                            true_interval_stats[aisle].append(order_interval)
                            
        true_stats = self._convert_to_prob_dict(true_stats)
        for key,value in true_interval_stats.items():
            value = self._convert_list_to_dist(value)
            true_interval_stats[key] = value
        
        true_stats = self._replace_nan(true_stats)
        return true_stats,true_interval_stats
    
    def save(self):
        stats = {str(k):{} for k in [1,2,3,4]}
        intervals = [(0,31),(0,8),(8,23),(23,31)]
        for interval in intervals:
            min_interval,max_interval = interval
            for cnt in [1,2,3,4]:
                params = {'valid_cnt':cnt,'min_interval':min_interval,'max_interval':max_interval}
                if cnt <= 3:
                    params.update({'comparator':'equal'}) 
                else:
                    params.update({'comparator':'greater_equal'})
                true_stats,_ = self.true_adjacent_stat(**params)
                key = f'cnt_{str(cnt)}_{str(min_interval)}_{str(max_interval)}'
                stats[str(cnt)][key] = true_stats
        path = os.path.join(TMP_PATH,'true_stats.pkl')
        pickle_save_load(path,stats,mode='save')
        
        fake_stats,fake_stats_interval = self.fake_adjacent_stat()
        for stat,stat_path in zip([fake_stats,fake_stats_interval],['fake_stats','fake_stats_interval']):
            path = os.path.join(TMP_PATH,f'{stat_path}.pkl')
            pickle_save_load(path,stat,mode='save')
    
    def load(self,path=None):
        if path is None:
            load_data = []
            for p in ['true_stats','fake_stats','fake_stats_interval']:
                path = os.path.join(TMP_PATH,f'{p}.pkl')
                load_data.append(pickle_save_load(path,mode='load'))
            return load_data

        load_data = pickle_save_load(path,mode='load')
        return load_data

def aisle_data_maker(agg_data,max_len,aisle_dept_dict,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_aisle','temporal_dict','aisle_data_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'aisle_data_dict_{suffix}'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        aisle_info_dim = data_dict[rand_key][0].shape[-1] - 1
        user_aisle = pickle_save_load(os.path.join(TMP_PATH,f'user_aisle_{suffix}'),mode='load')
        temp_data = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{suffix}'),mode='load')
        return user_aisle,data_dict,temp_data,aisle_info_dim
    
    user_aisle = []
    temporal_dict,data_dict = {},{}
    dpar = tqdm(agg_data.iterrows(),total=len(agg_data),desc='building aisle data for dataloader',
                dynamic_ncols=True,leave=False)
    for _,row in dpar:
        user_id,aisles,reorders = row['user_id'],row['aisle_id'],row['reordered']
        temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
        if mode != 'test' and row['eval_set'] == 'test':
            aisles = aisles[:-1]
            reorders = reorders[:-1]
            for idx,col_data in enumerate(temporal_cols):
                col_data = col_data[:-1]
                temporal_cols[idx] = col_data
        order_dow,order_hour,order_tz,order_days = temporal_cols
        order_dow = pad(np.roll(order_dow,-1)[:-1],max_len)
        order_hour = pad(np.roll(order_hour,-1)[:-1],max_len)
        order_tz = pad(np.roll(order_tz,-1)[:-1],max_len)
        order_days = np.roll(order_days,-1)[:-1]
        temporal_dict[user_id] = [order_dow,order_hour,order_tz,pad(order_days,max_len)]
        
        aisles,next_aisles = aisles[:-1],aisles[-1]
        next_aisles = next_aisles.split('_')
        orders = [aisle.split('_') for aisle in aisles]
        all_aisles = list(set(chain.from_iterable([aisle.split('_') for aisle in aisles])))
        reorders = reorders[:-1]
        reorders = [list(map(int,reorder.split('_'))) for reorder in reorders]
        
        for aisle in all_aisles:
            label = -1 if mode == 'test' else aisle in next_aisles 
            dept = aisle_dept_dict[int(aisle)]
            user_aisle.append([user_id,int(aisle),dept])
            
            order_sizes_ord = []
            in_order_ord  =[]
            index_order_ord = []
            index_order_ratio_ord = []
            cumsum_inorder_ratio_ord = []
            cumsum_inorder_ord = []
            cnt_ord = []
            cumsum_cnt_ratio_ord = []
            aisle_var_ord = []
            reorder_ratio_ord = []
            reorder_basket_ratio_ord = []
            reorder_total_ratio_ord = []
            cumsum_inorder,total_cnts,total_sizes,total_reorder_sums = 0,0,0,0
            seen_aisles = set()
            for idx,(order_aisle,reorder_aisle) in enumerate(zip(orders,reorders)):
                order_size = len(order_aisle)
                in_order = aisle in order_aisle
                order_set = list(unique_everseen(order_aisle))
                index_in_order = order_set.index(aisle) + 1 if in_order else 0
                index_in_order_ratio = index_in_order / len(order_set)
                total_orders = [1 if ele==aisle else 0 for ele in order_aisle]
                cumsum_inorder += in_order
                cumsum_inorder_ratio = cumsum_inorder / (idx + 1)
                cur_cnt = sum(total_orders)
                total_cnts += cur_cnt
                cumsum_cnt_ratio = total_cnts / (idx + 1)
                reorder_total = sum(reorder_aisle)
                total_reorder_sums += reorder_total
                reorder_ratio = total_reorder_sums / (idx + 1)
                reorder_basket_ratio = reorder_total / order_size
                cur_seens = seen_aisles & set(order_aisle)
                aisle_var_ord.append(len(cur_seens)/(idx+1))
                seen_aisles |= set(order_aisle)
                total_sizes += order_size
                reorder_total_ratio = total_reorder_sums / total_sizes

                in_order_ord.append(in_order)
                index_order_ord.append(index_in_order)
                index_order_ratio_ord.append(index_in_order_ratio)
                cumsum_inorder_ord.append(cumsum_inorder)
                cumsum_inorder_ratio_ord.append(cumsum_inorder_ratio)
                order_sizes_ord.append(order_size)
                cnt_ord.append(cur_cnt)
                cumsum_cnt_ratio_ord.append(cumsum_cnt_ratio)
                reorder_ratio_ord.append(reorder_ratio)
                reorder_basket_ratio_ord.append(reorder_basket_ratio)
                reorder_total_ratio_ord.append(reorder_total_ratio)
            
            next_in_order = np.roll(in_order_ord,-1)
            next_in_order[-1] = label
            
            aisle_info = np.stack([in_order_ord,np.array(index_order_ord)/145,index_order_ratio_ord,aisle_var_ord,reorder_ratio_ord,
                                   cumsum_inorder_ratio_ord,order_days/30,cumsum_cnt_ratio_ord,
                                   reorder_basket_ratio_ord,np.array(order_sizes_ord)/145,cnt_ord,reorder_total_ratio_ord,
                                   next_in_order],axis=1).astype(np.float16)
            aisle_info_dim = aisle_info.shape[1] - 1
            length = aisle_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,aisle_info_dim+1],dtype=np.float16)
            aisle_info = np.concatenate([aisle_info,paddings])
            
            data_dict[(user_id,int(aisle))] = (aisle_info,length)
            
    user_aisle = np.array(user_aisle)
    
    save_data  = [user_aisle,temporal_dict,data_dict]
    for path,file in zip(save_path,save_data):
        path = os.path.join(TMP_PATH,path)
        pickle_save_load(path,file,mode='save') 	
    
    return user_aisle,data_dict,temporal_dict,aisle_info_dim

def aisle_dataloader(inp,
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
        split_outputs = np.split(batch,batch.shape[1],axis=1)
        users,aisles,depts = list(map(np.squeeze,split_outputs))
        dows,hours,tzs,days = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,data_len = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(data_len)
        temporals = [dows,hours,tzs,days]
        yield full_batch,batch_lengths,temporals,users-1,aisles-1,depts-1

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()
class AisleTrainer(Trainer):
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
        self.dataloader = aisle_dataloader
        self.data_maker = aisle_data_maker
        self.model = AisleLSTM
        self.attr = 'aisle'
        
        self.emb_list = ['user_id','aisle_id','department_id']
        self.max_index_info = [data[col].max() for col in self.emb_list] + [self.max_len]
        aisle_dept_dict = self.prod_data.set_index('aisle_id')['department_id'].to_dict()
        self.data_maker_dicts = [aisle_dept_dict]
    
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

    trainer = AisleTrainer(data,
                            products,
                            output_dim=50,
                            lagging=2,
                            learning_rate=0.002,
                            optim_option='adam',
                            batch_size=256,
                            warm_start=False,
                            early_stopping=2,
                            epochs=10,
                            eval_epoch=1)
    trainer.train(use_amp=False,ev='evaluation/')
    trainer.predict(save_name='user_aisle_pred',ev='evaluation/')

#%%
# import torch
# import numpy as np
# x = torch.rand(32,100,25)
# y = torch.randint(1,100,(32,1,1)).long()
# y = y.expand(-1,1,25)
# z = torch.gather(x,index=y-1,dim=1).squeeze()
# np.concatenate([np.random.rand(32,10),np.random.rand(31,10)]).shape

# preds = np.load('metadata/user_product_pred.npy')
# z = preds[:1000]
