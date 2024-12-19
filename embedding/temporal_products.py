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
from itertools import product
import lightgbm as lgb
from nn_model.product_lstm import ProductTemporalNet
from utils.utils import logger,pickle_save_load,TMP_PATH
from itertools import chain
from embedding.trainer import Trainer

convert_index_cuda = lambda x:torch.from_numpy(x).long().cuda()

def product_data_maker(data,max_len,prod_aisle_dict,prod_dept_dict,mode='train'):
    suffix = mode + '.pkl'
    save_path = [path+'_'+suffix for path in ['user_prod','product_data_dict']]
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
            orders = [product.split('_') for product in products]
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
    
    save_data  = [user_prod,data_dict]
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
        self.model = ProductTemporalNet
        self.model_name = self.model.__class__.__name__
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
    
    def evaluate_or_submit(self,mode='evaluate'):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'learning_rate': .02,
            'num_leaves': 32,
            'max_depth': 12,
            'feature_fraction': 0.35,
            'bagging_fraction': 0.9,
            'bagging_freq': 2,
        }
        if mode == 'evaluate':
            addr = 'evaluation'
        else:
            addr = ''
        suffix = [s+'.npy' for s in ['eval','pred']]
        prefix = ['user_product','user_aisle','user_reorder','aisle_prob','product_prob']
        files = ['_'.join(comb) for comb in product(prefix,suffix)]
        checks = [os.path.exists(os.path.join(f'metadata/{addr}',file)) for file in files]
        assert np.all(checks),'all the eval and pred file of both user_product and user_aisle must be saved'
        
        data = [np.load(os.path.join(f'metadata/{addr}',file)) for file in files]
        user_prod_eval,user_prod_pred,user_aisle_eval,user_aisle_pred,user_reorder_eval,user_reorder_pred,\
        aisle_prob_eval,aisle_prob_pred,product_prob_eval,product_prob_pred = data
        
        nmf_item_emb = np.load('metadata/nmf_item_emb.npy')
        nmf_item_feat = [f'item{i}' for i in range(24)]
        nmf_item_emb = pd.DataFrame(nmf_item_emb,columns=['product_id']+nmf_item_feat)
        
        user_reorder_eval = user_reorder_eval[:,:-1]
        user_aisle_eval = user_aisle_eval[:,:-1]
        user_prod_eval[:,1] -= 1;user_prod_pred[:,1] -= 1
        label = user_prod_eval[:,-1]
        
        user_prod_eval = np.delete(user_prod_eval,-1,axis=1)
        prod_feat_name = [f'prodf_{i}' for i in range(51)]
        user_prod_eval = pd.DataFrame(user_prod_eval,columns=['user_id','product_id']+prod_feat_name)
        user_prod_eval['aisle_id'] = user_prod_eval['product_id'].map(self.prod_aisle_dict)
        aisle_feat_name = [f'aislef_{i}' for i in range(51)]
        user_aisle_eval = pd.DataFrame(user_aisle_eval,columns=['user_id','aisle_id']+aisle_feat_name)
        reorder_feat_name = [f'reorderf_{i}' for i in range(51)]
        user_reorder_eval = pd.DataFrame(user_reorder_eval,columns=['user_id']+reorder_feat_name)
        aisle_prob_feat = [f'aisle_prob_{i}' for i in range(2)]
        aisle_prob_eval = pd.DataFrame(aisle_prob_eval,columns=['user_id','aisle_id']+aisle_prob_feat)
        prod_prob_feat = [f'prod_prob_{i}' for i in range(2)]
        product_prob_eval = pd.DataFrame(product_prob_eval,columns=['user_id','product_id']+prod_prob_feat)
        data_tr = user_prod_eval.merge(user_aisle_eval,how='left',on=['user_id','aisle_id']).merge(nmf_item_emb,how='left',
                                    on=['product_id']).merge(user_reorder_eval,how='left',on=['user_id']).merge(aisle_prob_eval,
                                    how='left',on=['user_id','aisle_id']).merge(product_prob_eval,how='left',on=['user_id','product_id'])
        del data_tr['aisle_id'],data_tr['user_id'],data_tr['product_id']
        data_tr = np.array(data_tr).astype(np.float32)
        
        user_prod_pred = pd.DataFrame(user_prod_pred,columns=['user_id','product_id']+prod_feat_name)
        user_prod_pred['aisle_id'] = user_prod_pred['product_id'].map(self.prod_aisle_dict)
        user_aisle_pred = pd.DataFrame(user_aisle_pred,columns=['user_id','aisle_id']+aisle_feat_name)
        user_reorder_pred = pd.DataFrame(user_reorder_pred,columns=['user_id']+reorder_feat_name)
        aisle_prob_pred = pd.DataFrame(aisle_prob_pred,columns=['user_id','aisle_id']+aisle_prob_feat)
        product_prob_pred = pd.DataFrame(product_prob_pred,columns=['user_id','product_id']+prod_prob_feat)
        data_eval = user_prod_pred.merge(user_aisle_pred,how='left',on=['user_id','aisle_id']).merge(nmf_item_emb,how='left',
                                        on=['product_id']).merge(user_reorder_pred,how='left',on=['user_id']).merge(aisle_prob_pred,
                                        how='left',on=['user_id','aisle_id']).merge(product_prob_pred,how='left',on=['user_id','product_id'])
        del data_eval['aisle_id']

        user_prod_preds = np.array(data_eval[['user_id','product_id']]).astype(np.int32)
        data_eval = np.array(data_eval.iloc[:,2:])
        
        x_train = lgb.Dataset(data_tr,label=label)
        model = lgb.train(params,x_train,num_boost_round=500)
        predictions = model.predict(data_eval)
        predictions = np.concatenate([user_prod_preds,predictions.reshape(-1,1)],axis=1)
        predictions = pd.DataFrame(predictions,columns=['user_id','product_id','predictions'])
        if mode == 'submit':
            predictions.to_csv('metadata/user_product_prob.csv',index=False)
            return predictions
        
        lagging = self.lagging if np.sign(self.lagging) < 0 else -self.lagging
        agg_data = self.build_agg_data('product_id')
        agg_data_te = agg_data[agg_data['eval_set']=='test']
        rows = tqdm(agg_data_te.iterrows(),total=agg_data_te.shape[0],desc='collecting label for evaluation')
        user_prods = []
        for _,row in rows:
            user,prod = row['user_id'],row['product_id']
            reorder = row['reordered']
            prod = prod[lagging]
            reorder = reorder[lagging]
            reorder = list(map(int,reorder.split('_')))
            products = np.array(list(map(int,prod.split('_'))))
            users = np.array([user] * len(products))
            label = np.array([1] * len(products))
            if sum(reorder) == 0:
                products = np.append(products,0)
                users = np.append(users,user)
                label = np.append(label,1)
            user_prod = np.stack([users,products,label],axis=1)
            user_prods.append(user_prod)
        user_prods = np.concatenate(user_prods)
        user_prods = pd.DataFrame(user_prods,columns=['user_id','product_id','label'])
        predictions = user_prods.merge(predictions,how='outer',on=['user_id','product_id'])
        predictions[['label','predictions']] = predictions[['label','predictions']].fillna(0)
        y_true,y_pred = np.array(predictions['label']),np.array(predictions['predictions'])
        
        from sklearn.metrics import log_loss
        loss = log_loss(y_true,y_pred)
        logger.info(f'lightgbm prediction loss:{loss:.05f}')


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
    # product_trainer.train(use_amp=False,ev='')
    # product_trainer.predict(save_name='user_product_pred',ev='')
    product_trainer.evaluate_or_submit(mode='submit')



#%%

