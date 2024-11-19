# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:50:57 2024

@author: congx
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from utils.utils import pad
from create_merged_data import data_processing

def make_data(data,max_len,mode='train'):
    user_aisle = []
    temporal_dict,data_dict = {},{}
    dpar = tqdm(data.iterrows(),total=len(data),desc='building aisle data for dataloader',
                dynamic_ncols=True,leave=False)
    for _,row in dpar:
        user_id,aisles = row['user_id'],row['aisle_id']
        temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
        if mode != 'test' and row['eval_set'] == 'test':
            aisles = aisles[:-1]
            for idx,col_data in enumerate(temporal_cols):
                col_data = col_data[:-1]
                temporal_cols[idx] = col_data
        order_dow,order_hour,order_tz,order_days = temporal_cols
        order_dow = pad(np.roll(order_dow,-1)[:-1],max_len)
        order_hour = pad(np.roll(order_hour,-1)[:-1],max_len)
        order_tz = pad(np.roll(order_tz,-1)[:-1],max_len)
        order_days = np.roll(order_days,-1)[:-1]
        temporal_dict[user_id] = [order_dow.astype(np.int8),order_hour.astype(np.int8),order_tz.astype(np.int8)]
        
        aisles,next_aisles = aisles[:-1],aisles[-1]
        orders = [aisle.split('_') for aisle in aisles]
        all_aisles = list(set(chain.from_iterable([aisle.split('_') for aisle in aisles])))
        
        for aisle in all_aisles:
            label = -1 if mode == 'test' else aisle in next_aisles 
            label_cnt = -1 if mode == 'test' else sum([1 for ele in next_aisles if ele==aisle])
            user_aisle.append([user_id,int(aisle)])
            
            order_sizes_ord = []
            in_order_ord  =[]
            index_order_ord = []
            index_order_ratio_ord = []
            avg_index_ord = []
            cnt_ratio_ord = []
            avg_index_ratio_ord = []
            reorder_ratio_ord = []
            cumsum_inorder_ratio_ord = []
            cumsum_inorder_ord = []
            cnt_ord = []
            cnt_ratio_ord = []
            cumsum_cnt_ratio_ord = []
            cumsum_cnt_total_ratio_ord = []
            cumsum_inorder,total_cnts,total_cnts_log,total_sizes = 0,0,0,0
            seen_aisles = set()
            for idx,order_aisle in enumerate(orders):
                order_size = len(order_aisle)
                in_order = aisle in order_aisle
                index_in_order = order_aisle.index(aisle) if in_order else 0
                index_in_order_ratio = index_in_order / order_size
                total_orders = [idx+1 for idx,ele in enumerate(order_aisle) if ele==aisle]
                avg_pos = np.mean(total_orders) if in_order else 0
                avg_pos_ratio = avg_pos / order_size
                reorder = set(order_aisle) & seen_aisles
                seen_aisles |= set(order_aisle)
                reorder_ratio = len(reorder) / len(seen_aisles)
                cumsum_inorder += in_order
                cumsum_inorder_ratio = cumsum_inorder / (idx + 1)

                in_order_ord.append(in_order)
                index_order_ord.append(index_in_order)
                index_order_ratio_ord.append(index_in_order_ratio)
                avg_index_ord.append(avg_pos)
                avg_index_ratio_ord.append(avg_pos_ratio)
                reorder_ratio_ord.append(reorder_ratio)
                cumsum_inorder_ord.append(cumsum_inorder)
                cumsum_inorder_ratio_ord.append(cumsum_inorder_ratio)
                
                cur_cnt = len(total_orders)
                cur_cnt_ratio = cur_cnt / order_size
                total_cnts_log += np.log1p(cur_cnt)
                cumsum_cnt_ratio = total_cnts_log / (idx + 1)
                total_sizes += order_size
                total_cnts += cur_cnt
                cumsum_cnt_total_ratio = total_cnts / total_sizes
                
                order_sizes_ord.append(order_size)
                cnt_ord.append(cur_cnt)
                cnt_ratio_ord.append(cur_cnt_ratio)
                cumsum_cnt_ratio_ord.append(cumsum_cnt_ratio)
                cumsum_cnt_total_ratio_ord.append(cumsum_cnt_total_ratio)
            
            next_in_order = np.roll(in_order_ord,-1)
            next_in_order[-1] = label
            next_order_size = np.log1p(np.roll(cnt_ord,-1))
            next_order_size[-1] = np.log1p(label_cnt)
            
            aisle_info = np.stack([in_order_ord,np.array(index_order_ord)/145,index_order_ratio_ord,np.array(avg_index_ord)/100,
                                   avg_index_ratio_ord,reorder_ratio_ord,cumsum_inorder_ratio_ord,order_days/30,
                                   next_in_order],axis=1).astype(np.float16)
            aisle_info_dim = aisle_info.shape[1] - 1
            length = aisle_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,aisle_info_dim+1],dtype=np.float16)
            aisle_info = np.concatenate([aisle_info,paddings])
            
            aisle_cnt_info = np.stack([np.array(order_sizes_ord)/145,np.log1p(cnt_ord),cnt_ratio_ord,cumsum_cnt_ratio_ord,
                                       cumsum_cnt_total_ratio_ord,next_order_size],axis=1).astype(np.float16)
            aisle_cnt_dim = aisle_cnt_info.shape[1] - 1
            length = aisle_cnt_info.shape[0]
            padded_len = max_len - length
            paddings = np.zeros([padded_len,aisle_cnt_dim+1],dtype=np.float16)
            aisle_cnt_info = np.concatenate([aisle_cnt_info,paddings])
            
            data_dict[(user_id,int(aisle))] = (aisle_info,aisle_info_dim,aisle_cnt_info,aisle_cnt_dim)
    user_aisle = np.array(user_aisle)
    
    return user_aisle,data_dict,temporal_dict

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
        users,aisles = list(map(np.squeeze,split_outputs))
        dows,hours,tzs = zip(*list(map(temp_dict.get,users)))
        keys = np.split(batch,batch.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data_inorder,inorder_len,data_cnt,cnt_len = zip(*list(map(data_dict.get,keys)))
        full_batch_inorder,full_batch_cnt = np.stack(data_inorder),np.stack(data_cnt)
        inorder_len,cnt_len = list(inorder_len),list(cnt_len)
        temporals = [dows,hours,tzs]
        yield full_batch_inorder,inorder_len,full_batch_cnt,cnt_len,temporals,users-1,aisles-1

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    
    np.random.seed(9999)
    TEST_LENGTH = 7000000
    start_index = np.random.randint(0,data.shape[0]-TEST_LENGTH)
    end_index = start_index + TEST_LENGTH
    
    data_ = data.iloc[start_index:end_index]
    agg_data = data_processing(data_,save=False)
    z = make_data(agg_data,100,mode='train')
    

#%%
# import torch
# import pandas as pd
# import numpy as np
# data = pd.read_pickle('data/tmp/user_product_info.csv')
# z = data.iloc[:1000]
# # checkpoint = torch.load('checkpoint/ProdLSTM_best_checkpoint.pth')
# # orders = pd.read_csv('data/orders_info.csv')
# np.stack([[1,2,3],np.array([4,5,6])],axis=1)
dl = aisle_dataloader(z[0],z[1],z[2])
for batch in dl:
    break
#%%
