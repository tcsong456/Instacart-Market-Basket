# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:42:30 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import torch
import warnings
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm
from nn_model.lstm import ProdLSTM
from utils.utils import Timer,logger,pad
from torch.cuda.amp import autocast,GradScaler
from utils.loss import NextBasketLoss,SeqLogLoss
from itertools import chain
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
        
        z = data.groupby('user_id').apply(lambda x:x['eval_set'].iloc[-1])
        z.name = 'eval_set'
        
        r = r.merge(x,how='left',on='user_id').merge(z,how='left',on='user_id').reset_index()
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

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()

def model_data(data,max_len,prod_aisle_dict,prod_dept_dict,mode='train'):
    
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_prod','temporal_dict','data_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'data_dict_{mode}.pkl'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        feature_dim = data_dict[rand_key][0].shape[-1] - 1
        user_product = pickle_save_load(os.path.join(TMP_PATH,f'user_prod_{mode}.pkl'),mode='load')
        temp_data = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{mode}.pkl'),mode='load')
        return user_product,data_dict,temp_data,feature_dim
        
    if mode == 'train':
        lagging = -2
    elif mode == 'test':
        lagging = -1
        
    base_info = []
    data_dict,temporal_info_dict = {},{}
    with tqdm(total=data.shape[0],desc='building user data for dataloader',dynamic_ncols=True,
              leave=False) as pbar:
        for _,row in data.iterrows():
            user,products = row['user_id'],row['product_id']
            reorders = row['reordered']
            products,next_products = products[:lagging],products[lagging]
            orders = [product.split('_') for product in products]
            reorders_ = reorders[:lagging]
            
            reorders_ = [list(map(int,reorder.split('_'))) for reorder in reorders_]
            all_products = [list(set(chain.from_iterable(product.split('_') for product in products)))][0]
            
            for product in all_products:
                label = product in next_products if mode == 'train' else -1
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
                
                order_dow = pad(np.roll(row['order_dow'],-1)[:lagging],max_len)
                order_hour = pad(np.roll(row['order_hour_of_day'],-1)[:lagging],max_len)
                tz = pad(np.roll(row['time_zone'],-1)[:lagging],max_len)
                temporal_info = [order_dow,order_hour,tz]
                temporal_info_dict[user] = [b for b in temporal_info]
                
                days_interval = np.roll(row['days_since_prior_order'],-1)[:lagging]
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

get_checkpoint_path = lambda model_name,seed:f'{model_name}_best_checkpoint_{seed}'
def trainer(data,
            prod_data,
            output_dim,
            emb_dim,
            eval_epoch=3,
            epochs=100,
            save_data=True,
            learning_rate=0.01,
            train_size=0.8,
            batch_size=32,
            seed=18330,
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
    data_tr = model_data(agg_data,max_len,prod_aisle_dict,prod_dept_dict,mode='train')
    user_prod,prod_feat_dict,temporal_dict,feat_dim = data_tr
    for key,value in temporal_dict.items():
        for i in range(len(value)):
            value[i] = torch.Tensor(value[i]).long().cuda()
        temporal_dict[key] = value
    
    up_tr,up_val = train_test_split(user_prod,train_size=train_size,random_state=seed)
    max_index_info = [data[col].max() for col in emb_list] + [max_len]
    
    temp_dim = sum([data[t].max()+1 for t in temp_list])
    input_dim = feat_dim + len(emb_list) * emb_dim + temp_dim
    model = ProdLSTM(input_dim,output_dim,emb_dim,*max_index_info,True).to('cuda')
    model_name = model.__class__.__name__
    optimizer = optimizer(model.parameters(),lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.1,steps_per_epoch=up_tr.shape[0]//batch_size,
                                                     epochs=10)
    
    loss_fn_tr = SeqLogLoss(eps=1e-7)
    loss_fn_te = NextBasketLoss(eps=1e-7)
    
    checkpoint_path = get_checkpoint_path(model_name,seed)
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
        # total_ref_loss = 0
        model.train()
        train_dl = product_dataloader(up_tr,max_len,prod_feat_dict,temporal_dict,batch_size,True,True)
        train_batch_loader = tqdm(train_dl,total=up_tr.shape[0]//batch_size,desc=f'training next order basket at epoch:{epoch}',
                                  dynamic_ncols=True,leave=False)
        
        for batch,batch_lengths,_,temps,*aux_info in train_batch_loader:
            batch,label = batch[:,:,:-1],batch[:,:,-1]
            label = torch.from_numpy(label).to('cuda')
            batch = torch.from_numpy(batch).cuda()
            temps = [torch.stack(temp) for temp in temps]
            aux_info = [convert_index_cuda(b) for b in aux_info]
            
            optimizer.zero_grad()
            if use_amp:
                scaler = GradScaler()
                with autocast():
                    preds = model(batch,batch_lengths,*aux_info,*temps)
                    loss = loss_fn_tr(preds,label,batch_lengths)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(batch,batch_lengths,*aux_info,*temps)
                loss = loss_fn_tr(preds,label,batch_lengths)
                loss.backward()
                optimizer.step()
            
            # next_basket_label = torch.from_numpy(next_basket_label).cuda()
            # ref_loss = loss_fn_te(preds,next_basket_label,batch_lengths)
            # total_ref_loss += ref_loss
            # avg_ref_loss = total_ref_loss / cur_iter

            cur_loss = loss.item()
            total_loss += cur_loss
            avg_loss = total_loss / cur_iter
                
            cur_iter += 1
            train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
            lr_scheduler.step()
        
        if epoch % eval_epoch == 0:
            model.eval()
            total_loss_val,ite = 0,1
            eval_dl = product_dataloader(up_val,max_len,prod_feat_dict,temporal_dict,128,False,False)
            eval_batch_loader = tqdm(eval_dl,total=up_val.shape[0]//128,desc='evaluating next order basket',
                                     leave=False,dynamic_ncols=True)
            
            with torch.no_grad():
                for batch,batch_lengths,next_basket_label,temps,*aux_info in eval_batch_loader:
                    batch = batch[:,:,:-1]
                    batch = torch.from_numpy(batch).cuda()
                    next_basket_label = torch.from_numpy(next_basket_label).cuda()
                    temps = [torch.stack(temp) for temp in temps]
                    aux_info = [convert_index_cuda(b) for b in aux_info]
                    
                    preds = model(batch,batch_lengths,*aux_info,*temps)
                    loss = loss_fn_te(preds,next_basket_label,batch_lengths)
                    
                    total_loss_val += loss.item()
                    avg_loss_val = total_loss_val / ite
                    ite += 1
                
                    eval_batch_loader.set_postfix(eval_loss=f'{avg_loss_val:.05f}')
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
                logger.info('early stopping is trggered,the model has stopped improving for 3 consecutive epochs')
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
    data_te = model_data(predict_data,max_len,aisle_dict,dept_dict,mode='test')
    user_prod,prod_feat_dict,temporal_dict,feat_dim = data_te
    for key,value in temporal_dict.items():
        for i in range(len(value)):
            value[i] = torch.Tensor(value[i]).long().cuda()
        temporal_dict[key] = value
    
    model = ProdLSTM(input_dim,output_dim,emb_dim,*max_index_info,True).to('cuda')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dl = product_dataloader(user_prod,max_len,prod_feat_dict,temporal_dict,128,False,False)
    test_batch_loader = tqdm(test_dl,total=user_prod.shape[0]//128,desc='predicting next reorder basket',
                             leave=False,dynamic_ncols=True)
    predictions = []
    with torch.no_grad():
        for batch,batch_lengths,_,temps,users,prods,aisles,depts in test_batch_loader:
            batch = batch[:,:,:-1]
            batch = torch.from_numpy(batch).cuda()
            temps = [torch.stack(temp) for temp in temps]
            aux_info = [convert_index_cuda(b) for b in [users,prods,aisles,depts]]
            
            preds = model(batch,batch_lengths,*aux_info,*temps)
            preds = torch.sigmoid(preds).cpu()
            index = torch.Tensor(batch_lengths).long().reshape(-1,1) - 1
            probs = torch.gather(preds,dim=1,index=index).numpy()
            
            users +=1; prods += 1
            user_product = np.stack([users,prods],axis=1)
            user_product_prob  = np.concatenate([user_product,probs],axis=1)
            predictions.append(user_product_prob)
    
    predictions = np.concatenate(predictions)
    os.makedirs('metadata',exist_ok=True)
    np.save('metadata/user_product_prob.npy',predictions)
    return predictions

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')

    # z = data.iloc[1000000:2000000]
    # agg_data = data_processing(data,True)
    # pre_data = data_for_training(agg_data,max_len=100)
        
    outputs = trainer(data,
                    products,
                    output_dim=100,
                    emb_dim=50,
                    eval_epoch=1,
                    epochs=100,
                    save_data=False,
                    learning_rate=0.005,
                    train_size=0.8,
                    batch_size=512,
                    seed=18330,
                    early_stopping=3,
                    use_amp=False,
                    warm_start=True,
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
# import  numpy as np
# import pickle
# from utils import Timer
# from torch import nn
# import torch
# from torch.nn import functional as F
# from operator import itemgetter
# with open('data/tmp/user_prod.pkl','rb') as f:
#     user_prod = pickle.load(f)


# x = torch.rand(100,32).to('cuda')
# with open('data/tmp/try.pkl','rb') as f:
    # pickle.dump(x,f)
    # x = pickle.load(f)
# x = np.array([1,2,3,4,5])

# with Timer(8):
#     for _ in range(1000):
#         np.hstack([x,np.zeros(100)])
# z = user_prod[:10000]

#%%
# with Timer(precision=3):
#     cnt = 0
#     for batch,batch_lengths,next_basket_label,temps,*aux_info in product_dataloader(user_prod,100,data_dict,temp_dict,512,True,True):
#         batch,label = batch[:,:,:-1],batch[:,:,-1]
#         label = torch.from_numpy(label).to('cuda')
#         batch = torch.from_numpy(batch).cuda()
#         temps = [torch.stack(temp) for temp in temps]
#         aux_info = [convert_index_cuda(b) for b in aux_info]
#         if cnt == 100:
#             break
#         cnt += 1

        #%%
# z = np.load('metadata/user_product_prob.npy')
