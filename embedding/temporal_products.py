# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:42:30 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import pickle
import torch
import warnings
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm
from nn_model.lstm import ProdLSTM
from utils.utils import Timer,logger,pad,pickle_save_load,TMP_PATH
from torch.cuda.amp import autocast,GradScaler
from utils.loss import NextBasketLoss,SeqLogLoss
from itertools import chain
from sklearn.model_selection import train_test_split
from create_merged_data import data_processing

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()

def make_data(data,max_len,prod_aisle_dict,prod_dept_dict,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_prod','temporal_dict','data_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'data_dict_{suffix}'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        feature_dim = data_dict[rand_key][0].shape[-1] - 1
        user_product = pickle_save_load(os.path.join(TMP_PATH,f'user_prod_{suffix}'),mode='load')
        temp_data = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{suffix}'),mode='load')
        return user_product,data_dict,temp_data,feature_dim
        
    base_info = []
    data_dict,temporal_info_dict = {},{}
    with tqdm(total=data.shape[0],desc='building product data for dataloader',dynamic_ncols=True,
              leave=False) as pbar:
        for _,row in data.iterrows():
            user,products = row['user_id'],row['product_id']
            reorders = row['reordered']
            temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
            if row['eval_set'] == 'test' and mode != 'test':
                products = products[:-1]
                for idx,col_data in enumerate(temporal_cols):
                    col_data = col_data[:-1]
                    temporal_cols[idx] = col_data
            order_dow,order_hour,order_tz,order_days = temporal_cols
            products,next_products = products[:-1],products[-1]
            orders = [product.split('_') for product in products]
            reorders_ = reorders[:-1]
            
            reorders_ = [list(map(int,reorder.split('_'))) for reorder in reorders_]
            all_products = [list(set(chain.from_iterable(product.split('_') for product in products)))][0]
            
            for product in all_products:
                label = -1 if mode == 'test' else product in next_products
                aisle = prod_aisle_dict[int(product)]
                dept = prod_dept_dict[int(product)]
                base_info.append([user,int(product),aisle,dept])
                
                reorder_cnt = 0
                total_reorders,total_sizes = 0,0
                in_order_ord = []
                index_ord = []
                index_ratio_ord = []
                reorder_prod_ord = []
                reorder_prod_ratio_ord = []
                order_size_ord = []
                reorder_ord = []
                reorder_ratio_ord = []
                all_reorder_ratio_ord = []
                reorder_basket_ord = []
                for idx,(order,reorder) in enumerate(zip(orders,reorders_)):
                    in_order = int(product in order)
                    index_in_order = order.index(product) + 1 if in_order else 0
                    order_size = len(order)
                    index_order_ratio = index_in_order / order_size
                    reorder_cnt += int(in_order)
                    reorder_ratio_cum = reorder_cnt / (idx+1)
                    reorder_size = sum(reorder)
                    reorder_ratio = reorder_size / order_size
                    total_reorders += reorder_size
                    total_sizes += order_size
                    reorder_tendency = total_reorders / total_sizes
                    reorder_by_basket = total_reorders / (idx + 1)
                    
                    in_order_ord.append(in_order)
                    order_size_ord.append(order_size)
                    index_ord.append(index_in_order)
                    index_ratio_ord.append(index_order_ratio)
                    reorder_prod_ord.append(reorder_cnt)
                    reorder_prod_ratio_ord.append(reorder_ratio_cum)
                    reorder_ord.append(reorder_size)
                    reorder_ratio_ord.append(reorder_ratio)
                    all_reorder_ratio_ord.append(reorder_tendency)
                    reorder_basket_ord.append(reorder_by_basket)
                
                next_order_label = np.roll(in_order_ord,-1).reshape(-1,1)
                next_order_label[-1] = label
                
                order_dow = pad(np.roll(order_dow,-1)[:-1],max_len)
                order_hour = pad(np.roll(order_hour,-1)[:-1],max_len)
                tz = pad(np.roll(order_tz,-1)[:-1],max_len)
                temporal_info = [order_dow,order_hour,tz]
                temporal_info_dict[user] = [b for b in temporal_info]
                
                days_interval = np.roll(order_days,-1)[:-1]
                prod_info = np.stack([in_order_ord,np.array(order_size_ord)/145,np.array(index_ord)/145,index_ratio_ord,
                                      np.array(reorder_prod_ord)/100,reorder_prod_ratio_ord,np.array(reorder_ord)/130,reorder_ratio_ord,
                                      all_reorder_ratio_ord,reorder_basket_ord,days_interval/30]).transpose()
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
        users,prods,aisles,depts = list(map(np.squeeze,split_outputs))
        dows,hours,tzs = zip(*list(map(temp_dict.get,users)))
        keys = batch[:,:2]
        keys = np.split(keys,keys.shape[0],axis=0)
        keys = list(map(tuple,(map(np.squeeze,keys))))
        data,length = zip(*list(map(data_dict.get,keys)))
        full_batch = np.stack(data)
        batch_lengths = list(length)
        temporals = [dows,hours,tzs]
        yield full_batch,batch_lengths,temporals,users-1,prods-1,aisles-1,depts-1

get_checkpoint_path = lambda model_name:f'{model_name}_best_checkpoint.pth'
def trainer(data,
            prod_data,
            output_dim,
            emb_dim,
            eval_epoch=3,
            epochs=100,
            save_data=True,
            learning_rate=0.01,
            lagging=1,
            batch_size=32,
            early_stopping=5,
            use_amp=True,
            warm_start=False,
            scheduler_option='reduce',
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
    agg_data_tr,agg_data_te = agg_data[agg_data['eval_set']=='train'],agg_data[agg_data['eval_set']=='test']
    agg_data_tr = agg_data_tr[agg_data_tr['product_id'].map(lambda x:len(x)-lagging>=2)]
    agg_data_te = agg_data_te[agg_data_te['product_id'].map(lambda x:len(x)-(lagging+1)>=2)]
    agg_data = pd.concat([agg_data_tr,agg_data_te])
    del agg_data_tr,agg_data_te
    gc.collect()
    user_prod,prod_feat_dict,temporal_dict,feat_dim = make_data(agg_data,max_len,prod_aisle_dict,prod_dept_dict,mode='train')

    for key,value in temporal_dict.items():
        for i in range(len(value)):
            value[i] = torch.Tensor(value[i]).long().cuda()
        temporal_dict[key] = value

    max_index_info = [data[col].max() for col in emb_list] + [max_len]
    
    temp_dim = sum([data[t].max()+1 for t in temp_list])
    input_dim = feat_dim + len(emb_list) * emb_dim + temp_dim
    model = ProdLSTM(input_dim,output_dim,emb_dim,*max_index_info).to('cuda')

    model_name = model.__class__.__name__
    optimizer = optimizer(model.parameters(),lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=0,factor=0.2,verbose=True)
    
    loss_fn_tr = SeqLogLoss(lagging=lagging,eps=1e-7)
    loss_fn_te = NextBasketLoss(lagging=lagging,eps=1e-7)
    
    checkpoint_path = get_checkpoint_path(model_name)
    checkpoint_path = f'checkpoint/{checkpoint_path}'
    if warm_start:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['best_epoch']
        best_loss = checkpoint['best_loss']
        logger.info(f'warm starting training from best epoch:{start_epoch} and best loss:{best_loss:.5f}')
    else:
        start_epoch = 0
        best_loss = np.inf
    
    no_improvement = 0
    for epoch in range(start_epoch,epochs):
        total_loss,cur_iter = 0,1
        model.train()
        train_dl = product_dataloader(user_prod,max_len,prod_feat_dict,temporal_dict,batch_size,True,True)
        train_batch_loader = tqdm(train_dl,total=user_prod.shape[0]//batch_size,desc=f'training next order basket at epoch:{epoch}',
                                  dynamic_ncols=True,leave=False)
        
        for batch,batch_lengths,temps,*aux_info in train_batch_loader:
            batch,label = batch[:,:,:-1],batch[:,:,-1]
            label = torch.from_numpy(label).to('cuda')
            batch = torch.from_numpy(batch).cuda()
            temps = [torch.stack(temp) for temp in temps]
            aux_info = [convert_index_cuda(b) for b in aux_info]
            
            optimizer.zero_grad()
            if use_amp:
                scaler = GradScaler()
                with autocast():
                    preds = model(batch,*aux_info,*temps)
                    loss = loss_fn_tr(preds,label,batch_lengths)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(batch,*aux_info,*temps)
                loss = loss_fn_tr(preds,label,batch_lengths)
                loss.backward()
                optimizer.step()

            cur_loss = loss.item()
            total_loss += cur_loss
            avg_loss = total_loss / cur_iter
                
            cur_iter += 1
            train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
        
        if epoch % eval_epoch == 0:
            model.eval()
            total_loss_val,ite = 0,1
            eval_dl = product_dataloader(user_prod,max_len,prod_feat_dict,temporal_dict,1024,False,False)
            eval_batch_loader = tqdm(eval_dl,total=user_prod.shape[0]//1024,desc='evaluating next order basket',
                                     leave=False,dynamic_ncols=True)
            
            with torch.no_grad():
                for batch,batch_lengths,temps,*aux_info in eval_batch_loader:
                    batch,label = batch[:,:,:-1],batch[:,:,-1]
                    batch = torch.from_numpy(batch).cuda()
                    label = torch.from_numpy(label).to('cuda')
                    temps = [torch.stack(temp) for temp in temps]
                    aux_info = [convert_index_cuda(b) for b in aux_info]
                    
                    preds = model(batch,*aux_info,*temps)
                    loss = loss_fn_te(preds,label,batch_lengths)
                    
                    total_loss_val += loss.item()
                    avg_loss_val = total_loss_val / ite
                    ite += 1
                    eval_batch_loader.set_postfix(eval_loss=f'{avg_loss_val:.05f}')
            lr_scheduler.step(avg_loss_val)
            
            if avg_loss_val < best_loss:
                best_loss = avg_loss_val
                checkpoint = {'best_epoch':epoch,
                              'best_loss':best_loss,
                              'model_state_dict':model.state_dict(),
                              'optimizer_state_dict':optimizer.state_dict()}
                os.makedirs('checkpoint',exist_ok=True)
                torch.save(checkpoint,checkpoint_path)
                no_improvement = 0
            else:
                no_improvement += 1
                
            if no_improvement == early_stopping:
                logger.info('early stopping is trggered,the model has stopped improving')
                return agg_data,input_dim,max_len,prod_aisle_dict,prod_dept_dict,max_index_info,checkpoint_path
    return agg_data,input_dim,max_len,prod_aisle_dict,prod_dept_dict,max_index_info,checkpoint_path
                
def predict(
            agg_data,
            input_dim,
            max_len,
            aisle_dict,
            dept_dict,
            max_index_info,
            checkpoint_path,
            output_dim,
            emb_dim
            ):
    predict_data = agg_data[agg_data['eval_set']=='test']
    data_te = make_data(predict_data,max_len,aisle_dict,dept_dict,mode='test')
    user_prod,prod_feat_dict,temporal_dict,feat_dim = data_te
    for key,value in temporal_dict.items():
        for i in range(len(value)):
            value[i] = torch.Tensor(value[i]).long().cuda()
        temporal_dict[key] = value

    model = ProdLSTM(input_dim,output_dim,emb_dim,*max_index_info).to('cuda')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dl = product_dataloader(user_prod,max_len,prod_feat_dict,temporal_dict,1024,False,False)
    test_batch_loader = tqdm(test_dl,total=user_prod.shape[0]//1024,desc='predicting next reorder basket',
                             leave=False,dynamic_ncols=True)
    predictions = []
    with torch.no_grad():
        for batch,batch_lengths,temps,users,prods,aisles,depts in test_batch_loader:
            batch = batch[:,:,:-1]
            batch = torch.from_numpy(batch).cuda()
            temps = [torch.stack(temp) for temp in temps]
            aux_info = [convert_index_cuda(b) for b in [users,prods,aisles,depts]]
            
            preds = model(batch,*aux_info,*temps)
            preds = torch.sigmoid(preds).cpu()
            index = torch.Tensor(batch_lengths).long().reshape(-1,1) - 1
            probs = torch.gather(preds,dim=1,index=index).numpy()
            
            users +=1; prods += 1
            user_product = np.stack([users,prods],axis=1)
            user_product_prob  = np.concatenate([user_product,probs],axis=1)
            predictions.append(user_product_prob)
    
    predictions = np.concatenate(predictions).astype(np.float32)
    os.makedirs('metadata',exist_ok=True)
    pred_path = 'metadata/user_product_prob.npy'
    np.save(pred_path,predictions)
    logger.info(f'predictions saved to {pred_path}')
    return predictions

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')
    # z = data.iloc[1000000:2000000]
        
    outputs = trainer(data,
                    products,
                    output_dim=100,
                    emb_dim=50,
                    eval_epoch=1,
                    epochs=1,
                    save_data=False,
                    learning_rate=0.002,
                    lagging=1,
                    batch_size=512,
                    early_stopping=2,
                    use_amp=False,
                    warm_start=False,
                    optim_option='adam')
    predict(*outputs,
            output_dim=100,
            emb_dim=50)
    
        
    # from collections import ChainMap
    # from functools import partial
    # import multiprocessing
    # prod_aisle_dict = products.set_index('product_id')['aisle_id'].to_dict()
    # prod_dept_dict = products.set_index('product_id')['department_id'].to_dict()
    
    # agg_data = data_processing(z,save=False)
    # func = partial(data_for_training,max_len=100,prod_aisle_dict=prod_aisle_dict,prod_dept_dict=prod_dept_dict)
    # chunks = []
    # cpu_cnt = 8
    # chunk_size = agg_data.shape[0] // cpu_cnt
    # for i in range(cpu_cnt-1):
    #     start = i * chunk_size
    #     end = (i+1) * chunk_size
    #     chunks.append(agg_data.iloc[start:end])
    # chunks.append(agg_data.iloc[end:])
    # with Timer():
    #     with multiprocessing.Pool(cpu_cnt) as pool:
    #         results = pool.map(func,chunks)
    #         user_prod,data_dict,temporal_info_dict,feature_dims = zip(*results)
    #         user_prod = np.concatenate(user_prod)
    #         print(user_prod,user_prod.shape)
    #         temporal_info_dict = dict(ChainMap(*temporal_info_dict))
    #         print(temporal_info_dict)
    #         data_dict = dict(ChainMap(*data_dict))
    #         print(data_dict)
    #         feature_dim = feature_dims[0]
    #         print(feature_dim)



#%%
np.random.seed(9999)
TEST_LENGTH = 7000000
start_index = np.random.randint(0,data.shape[0]-TEST_LENGTH)
start_index