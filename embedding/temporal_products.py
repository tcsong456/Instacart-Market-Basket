# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:42:30 2024

@author: congx
"""
import os
import sys
import pickle
import torch
import warnings
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm
from utils import Timer,logger,SeqLogLoss,pad
from itertools import chain
from nn_model.lstm import ProdLSTM
from sklearn.model_selection import train_test_split

TMP_PATH = 'data/tmp'

def data_processing(data,save=False):
    os.makedirs(TMP_PATH,exist_ok=True)
    path = os.path.join(TMP_PATH,'user_product_info.csv')
    try:
        r = pd.read_pickle(path)
    except FileNotFoundError:
        logger.info('building user data')
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
            x = x.reset_index().sort_values(['user_id','order_number']).groupby('user_id')[col].apply(list)
            grouped_attr.append(x)
        x = pd.concat(grouped_attr,axis=1)
        r = r.merge(x,how='left',on='user_id').reset_index()
        if save:
            r.to_pickle(path)
    return r


def pickle_save_load(path,data=None,mode='save'):
    if mode == 'save':
        assert data is not None,'data must be provided when it is in save mode'
        with open(path,'wb') as f:
            pickle.dump(data,f)
    elif mode == 'load':
        with open(path,'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise KeyError(f'{mode} is invalid mode')

def data_for_training(data,max_len,prod_aisle_dict,prod_dept_dict):
    
    save_path = ['user_prod.pkl','temporal_dict.pkl','data_dict.pkl']
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,'data_dict.pkl'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        feature_dim = data_dict[rand_key][0].shape[-1] - 1
        user_product = pickle_save_load(os.path.join(TMP_PATH,'user_prod.pkl'),mode='load')
        temp_data = pickle_save_load(os.path.join(TMP_PATH,'temporal_dict.pkl'),mode='load')
        return user_product,data_dict,temp_data,feature_dim
        
    base_info = []
    data_dict,temporal_info_dict = {},{}
    
    with tqdm(total=data.shape[0],desc='building user data for dataloader') as pbar:
        for _,row in data.iterrows():
            user,products = row['user_id'],row['product_id']
            reorders = row['reordered']
            orders = [product.split('_') for product in products]
            products,next_products = products[:-1],products[-1]
            reorders_ = reorders[:-1]
            
            reorders_ = [list(map(int,reorder.split('_'))) for reorder in reorders_]
            all_products = [list(set(chain.from_iterable(product.split('_') for product in products)))][0]
            
            for product in all_products:
                label = product in next_products
                aisle = prod_aisle_dict[int(product)]
                dept = prod_dept_dict[int(product)]
                base_info.append([user,int(product),aisle,dept,label])
                
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
                
                next_order_label = np.roll(in_order_ord,-1).reshape(-1,1)
                
                order_dow = pad(np.roll(row['order_dow'],-1)[:-1],max_len)
                order_hour = pad(np.roll(row['order_hour_of_day'],-1)[:-1],max_len)
                tz = pad(np.roll(row['time_zone'],-1)[:-1],max_len)
                temporal_info_dict[user] = [order_dow,order_hour,tz]
                
                days_interval = np.roll(row['days_since_prior_order'],-1)[:-1]
                prod_info = np.stack([in_order_ord,order_size_ord,index_ord,index_ratio_ord,
                                      reorder_prod_ord,reorder_prod_ratio_ord,reorder_ord,reorder_ratio_ord,
                                      days_interval]).transpose()
                prod_info = np.concatenate([prod_info,next_order_label],axis=1).astype(np.float16)
                length = prod_info.shape[0]
                feature_dim = prod_info.shape[-1] - 1
                missing_seq = max_len - length
                if missing_seq > 0:
                    missing_data = np.zeros([missing_seq,prod_info.shape[1]],dtype=np.float16)
                    prod_info = np.concatenate([prod_info,missing_data])
                
                data_dict[(user,int(product))] = (prod_info,length)
            pbar.update(1)
    user_prod = np.stack(base_info)
    #
    save_data  = [user_prod,temporal_info_dict,data_dict]
    for path,file in zip(save_path,save_data):
        path = os.path.join(TMP_PATH,path)
        pickle_save_load(path,file,mode='save')
    
    return user_prod,data_dict,temporal_info_dict,feature_dim


def product_dataloader(inp,
                       max_len,
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
        users,prods,aisles,depts,labels = list(map(np.squeeze,split_outputs))
        dows,hours,tzs = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,length = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(length)
        temporals = [dows,hours,tzs]
        yield full_batch,batch_lengths,labels,temporals,users-1,prods-1,aisles-1,depts-1

convert_index_cuda = lambda x:torch.Tensor(x).long().to('cuda')

def trainer(data,
            prod_data,
            output_dim,
            emb_dim,
            loss_fn,
            save_data=True,
            learning_rate=0.01,
            train_size=0.8,
            batch_size=32,
            dropout=0.0,
            seed=18330,
            use_amp=True,
            optim_option='adam'):
    optimizer_ = optim_option.lower()
    if optimizer_ == 'sgd':
        optimizer = optim.SGD
    elif optimizer_ == 'adam':
        optimizer = optim.Adam
    elif optimizer_ == 'adamw':
        optimizer = optim.AdamW
    else:
        logger.warning(f'{optimizer_} is an invalid option for optimizer')
        sys.exit(1)
    
    max_len = data['order_number'].max()
    temp_list = ['order_dow','order_hour_of_day','time_zone']
    emb_list = ['user_id','product_id','aisle_id','department_id']
    prod_aisle_dict = prod_data.set_index('product_id')['aisle_id'].to_dict()
    prod_dept_dict = prod_data.set_index('product_id')['department_id'].to_dict()
    
    agg_data = data_processing(data,save=save_data)
    data_tr = data_for_training(agg_data,max_len,prod_aisle_dict,prod_dept_dict)
    user_prod,prod_feat_dict,temporal_dict,feat_dim = data_tr
    up_tr,up_te = train_test_split(user_prod,train_size=train_size,random_state=seed)
    max_index_info = [data[col].max() for col in emb_list] + [max_len]
    
    train_dl = product_dataloader(up_tr,max_len,prod_feat_dict,temporal_dict,batch_size,True,True)
    train_batch_loader = tqdm(train_dl,total=up_tr.shape[0]//batch_size,desc='training next order basket')
    
    temp_dim = sum([data[t].max()+1 for t in temp_list])
    input_dim = feat_dim + len(emb_list) * emb_dim + temp_dim
    model = ProdLSTM(input_dim,output_dim,emb_dim,*max_index_info,1,True,dropout).to('cuda')
    optimizer = optimizer(model.parameters(),lr=learning_rate)
    model.train()
    
    from torch.cuda.amp import autocast,GradScaler
    total_loss,cur_iter = 0,1
    scaler = GradScaler()
    for batch,batch_lengths,next_basket_label,temps,*aux_info in train_batch_loader:
        batch,label = batch[:,:,:-1],batch[:,:,-1]
        label = torch.from_numpy(label).to('cuda')
        batch = torch.from_numpy(batch).cuda()
        temps = [convert_index_cuda(temp) for temp in temps]
        aux_info = [convert_index_cuda(b) for b in aux_info]
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                preds = model(batch,batch_lengths,*aux_info,*temps)
                loss = loss_fn(preds,label,batch_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(batch,batch_lengths,*aux_info,*temps)
            loss = loss_fn(preds,label,batch_lengths)
            loss.backward()
            optimizer.step()
        
        cur_loss = loss.item()
        total_loss += cur_loss
        avg_loss = total_loss / cur_iter
            
        cur_iter += 1
        train_batch_loader.set_postfix(loss=f'{avg_loss:.05f}')

if __name__ == '__main__':
    # try:
    #     data = pd.read_csv('data/orders_info.csv')
    #     products = pd.read_csv('data/products.csv')
    # except FileNotFoundError as e:
    #     logger.warning(f"{e}! Please create the data using 'create_merged_data' file first")
    #     sys.exit(1)
    
    with Timer(precision=0):
        z = data.iloc[1000000:2000000]
        # agg_data = data_processing(data,True)
        # pre_data = data_for_training(agg_data,max_len=100)
        
        loss_fn = SeqLogLoss(eps=1e-7)
        trainer(data,
                products,
                100,
                50,
                loss_fn,
                save_data=False,
                train_size=0.8,
                batch_size=512,
                dropout=0.0,
                seed=18330,
                use_amp=True,
                optim_option='adam')



#%%
import  numpy as np
import pickle
from utils import Timer
from torch import nn
import torch
from torch.nn import functional as F
from operator import itemgetter
# with open('data/tmp/temporal_dict.pkl','rb') as f:
#     temp_dict = pickle.load(f)

# F.one_hot(torch.from_numpy(temp_dict[6243][1]).long(),num_classes=24)
# with Timer(8):
#     x = torch.randint(0,10,(128,25))
#     print(F.one_hot(x,num_classes=10))
    # for i in range(1,200000):
        # key = np.random.randint(1,206210)
        # x = temp_dict[i]
    # x = list(map(temp_dict.get,np.arange(1,1024)))
    # a,b,c = zip(*x)
    # print(np.array())

    # itemgetter(*np.arange(1,200000))(temp_dict)
    # user_prod[]
#%%
# d = {}
# d[(1,2)] = (np.random.rand(32,5),11)
# d[(3,4)] = (np.random.rand(32,5),12)
# d[(5,6)] = (np.random.rand(32,5),13)
# a,b = zip(*list(map(d.get,np.array([(1,2),(3,4),(5,6)]))))
# x = np.random.rand(32,5)
# a,*b = np.split(x,x.shape[1],axis=1)
# x = map(np.squeeze,a)

