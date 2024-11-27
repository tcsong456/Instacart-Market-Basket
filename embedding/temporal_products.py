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
from nn_model.product_lstm import ProdLSTM,ProdLSTMV1
from utils.utils import Timer,logger,pad,pickle_save_load,TMP_PATH
from torch.cuda.amp import autocast,GradScaler
from utils.loss import NextBasketLoss,SeqLogLoss
from itertools import chain
from sklearn.model_selection import train_test_split
from create_merged_data import data_processing
from embedding.trainer import Trainer

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()

def product_data_maker(data,max_len,prod_aisle_dict,prod_dept_dict,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_prod','product_data_dict','temporal_dict']]
    check_files = np.all([os.path.exists(os.path.join(TMP_PATH,file)) for file in save_path])
    temp_dict = pickle_save_load(os.path.join(TMP_PATH,f'temporal_dict_{suffix}'),mode='load')
    if check_files:
        logger.info('loading temporary data')
        data_dict = pickle_save_load(os.path.join(TMP_PATH,f'product_data_dict_{suffix}'),mode='load')
        keys = list(data_dict.keys())
        rand_index = np.random.randint(0,len(keys),1)[0]
        rand_key = keys[rand_index]
        feature_dim = data_dict[rand_key][0].shape[-1] - 1
        user_product = pickle_save_load(os.path.join(TMP_PATH,f'user_prod_{suffix}'),mode='load')
        return user_product,data_dict,temp_dict,feature_dim
        
    base_info = []
    data_dict = {}
    temp_dict = {}
    with tqdm(total=data.shape[0],desc='building product data for dataloader',dynamic_ncols=True,
              leave=False) as pbar:
        for _,row in data.iterrows():
            user,products = row['user_id'],row['product_id']
            reorders = row['reordered']
            temporal_cols = [row['order_dow'],row['order_hour_of_day'],row['time_zone'],row['days_since_prior_order']]
            # days_interval = row['days_since_prior_order']
            if row['eval_set'] == 'test' and mode != 'test':
                products = products[:-1]
                reorders = reorders[:-1]
                # days_interval = days_interval[:-1]
                for idx,col_data in enumerate(temporal_cols):
                    col_data = col_data[:-1]
                    temporal_cols[idx] = col_data
            order_dow,order_hour,order_tz,days_interval = temporal_cols
            order_dow = pad(np.roll(order_dow,-1)[:-1],max_len)
            order_hour = pad(np.roll(order_hour,-1)[:-1],max_len)
            order_tz = pad(np.roll(order_tz,-1)[:-1],max_len)
            days_interval = np.roll(days_interval,-1)[:-1]
            temp_dict[user] = [order_dow,order_hour,order_tz]
            
            products,next_products = products[:-1],products[-1]
            next_products = next_products.split('_')
            orders = [product.split('_') for product in products]
            reorders_,next_reorders = reorders[:-1],reorders[-1]
            
            reorders_ = [list(map(int,reorder.split('_'))) for reorder in reorders_]
            all_products = list(set(chain.from_iterable([product.split('_') for product in products])))
            
            for product in all_products:
                label = -1 if mode == 'test' else product in next_products
                aisle = prod_aisle_dict[int(product)]
                dept = prod_dept_dict[int(product)]
                base_info.append([user,int(product),aisle,dept])
                
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
                next_order_label[-1] = label
                
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
                data_dict[(user,int(product))] = (prod_info,length)
            
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
    
    save_data  = [user_prod,data_dict,temp_dict]
    for path,file in zip(save_path,save_data):
        path = os.path.join(TMP_PATH,path)
        pickle_save_load(path,file,mode='save') 	
    
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
        self.model = ProdLSTM
        self.data_maker = product_data_maker
        self.emb_list = ['user_id','product_id','aisle_id','department_id']
        self.max_index_info = [data[col].max()+1 for col in self.emb_list] + [self.max_len]
        prod_aisle_dict = self.prod_data.set_index('product_id')['aisle_id'].to_dict()
        prod_dept_dict = self.prod_data.set_index('product_id')['department_id'].to_dict()
        self.data_maker_dicts = [prod_aisle_dict,prod_dept_dict]
    
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
        return [user_prod,data_dict,temp_dict],prod_dim
    
    def train(self,use_amp=False,use_extra_params=False):
        core_info_tr,prod_dim = self.build_data_dl(mode='train')
        model,optimizer,lr_scheduler,start_epoch,best_loss,checkpoint_path = super().train(use_amp=use_amp)
        self.checkpoint_path = checkpoint_path
        
        if use_extra_params:
            aisle_param_path = 'data/tmp/user_aisle_param.pkl'
            aisle_param_dict = pickle_save_load(aisle_param_path,mode='load')
            emb_list = ['user_id','product_id','department_id']
            self.input_dim = self.temp_dim + prod_dim + len(emb_list) * 50 + 21
            max_index_info = [data[col].max() for col in emb_list] + [self.max_len]
            model = ProdLSTMV1(self.input_dim,self.output_dim,*max_index_info,aisle_param_dict).cuda()
            optimizer = self.optimizer(model.parameters(),lr=self.learning_rate)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=0,factor=0.2,verbose=True)
            
        no_improvement = 0
        for epoch in range(start_epoch,self.epochs):
            total_loss,cur_iter = 0,1
            model.train()
            train_dl = self.dataloader(*core_info_tr,self.batch_size,True,True)
            train_batch_loader = tqdm(train_dl,total=core_info_tr[0].shape[0]//self.batch_size,desc=f'training next product basket at epoch:{epoch}',
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
                        h,preds = model(batch,*temps)
                        loss = self.loss_fn_tr(preds,label,batch_lengths)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    h,preds = model(batch,*aux_info,*temps)
                    loss = self.loss_fn_tr(preds,label,batch_lengths)
                    loss.backward()
                    optimizer.step()

                cur_loss = loss.item()
                total_loss += cur_loss
                avg_loss = total_loss / cur_iter
                    
                cur_iter += 1
                train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
            
            if epoch % self.eval_epoch == 0:
                pred_embs_eval = []
                model.eval()
                total_loss_val,ite = 0,1
                eval_dl = self.dataloader(*core_info_tr,1024,False,False)
                eval_batch_loader = tqdm(eval_dl,total=core_info_tr[0].shape[0]//1024,desc='evaluating next order basket',
                                          leave=False,dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch,batch_lengths,temps,*aux_info in eval_batch_loader:
                        batch,label = batch[:,:,:-1],batch[:,:,-1]
                        batch = torch.from_numpy(batch).cuda()
                        label = torch.from_numpy(label).to('cuda')
                        temps = [torch.stack(temp) for temp in temps]
                        aux_info = [convert_index_cuda(b) for b in aux_info]
                        
                        h,preds = model(batch,*aux_info,*temps)
                        loss = self.loss_fn_te(preds,label,batch_lengths)
                        
                        total_loss_val += loss.item()
                        avg_loss_val = total_loss_val / ite
                        ite += 1
                        eval_batch_loader.set_postfix(eval_loss=f'{avg_loss_val:.05f}')
                        
                        pred_emb_val = self._collect_final_time_step(batch_lengths,aux_info,h,label)
                        pred_embs_eval.append(pred_emb_val)
                lr_scheduler.step(avg_loss_val)
                pred_embs_eval = np.concatenate(pred_embs_eval).astype(np.float32)
                
                if avg_loss_val < best_loss:
                    best_loss = avg_loss_val
                    checkpoint = {'best_epoch':epoch,
                                  'best_loss':best_loss,
                                  'model_state_dict':model.state_dict(),
                                  'optimizer_state_dict':optimizer.state_dict()}
                    os.makedirs('checkpoint',exist_ok=True)
                    torch.save(checkpoint,checkpoint_path)
                    np.save('metadata/user_product_eval.npy',pred_embs_eval)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                if no_improvement == self.early_stopping:
                    logger.info('early stopping is trggered,the model has stopped improving')
                    return
                


if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    products = pd.read_csv('data/products.csv')
    # z = data.iloc[1000000:2000000]
        
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
    product_trainer.train(use_amp=False,use_extra_params=False)
    product_trainer.predict(save_name='user_product_pred')
    # predict(*outputs,
    #         output_dim=100,
    #         emb_dim=50)
    
        
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
# import pandas as  pd
# import numpy as np
# import torch
# import pickle
# data = pd.read_csv('data/orders_info.csv')
# z = data.iloc[:100000]
# gb = z.groupby('order_id')
# dfs, order_ids = zip(*[(df, key) for key, df in gb])
# p = np.load('metadata/user_product_prob.npy')
# checkpoint = torch.load('checkpoint/ProdLSTM_best_checkpoint.pth')
# with open('data/tmp/user_prod_test.pkl','rb') as f:
#     user_prod = pickle.load(f)
