# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:39:09 2024

@author: congx
"""
import gc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from collections import defaultdict
from nn_model.bpr import BPREmbeddingModel
from utils.loss import BPRLoss
from utils.utils import logger,pickle_save_load
from sklearn.model_selection import train_test_split

def pos_pair_cnt(data):
    data = data[data['eval_set']!='test']
    order_products = data.groupby(['user_id','order_id','order_number'])['product_id'].apply(list).reset_index()
    order_products = order_products.sort_values(['user_id','order_number'])
    del order_products['order_number']
    shift_order_products = order_products.groupby('user_id')['product_id'].shift(-1)
    shift_order_products.name = 'shift_products'
    order_products = pd.concat([order_products,shift_order_products],axis=1)
    del data
    gc.collect()
    
    stats_cnt = defaultdict(int)
    rows = tqdm(order_products.iterrows(),total=order_products.shape[0],desc='collecting count stats for positive pairs')
    for _,row in rows:
        products,next_products = row['product_id'],row['shift_products']
        if np.isnan(next_products).any():
            continue
        for next_prod in next_products:
            for prod in products:
                stats_cnt[(next_prod,prod)] += 1
    pickle_save_load('data/temp/product_stats_cnt.pkl',mode='save')
    return order_products,stats_cnt

def pos_pair_stats(data,pos_cnt_dict,top=3):
    pos_pair = defaultdict(list)
    rows = tqdm(data.iterrows(),total=data.shape[0],desc='collecting positive pairs stats')
    for _,row in rows:
        products,next_products = row['product_id'],row['shift_products']
        if np.isnan(next_products).any():
            continue
        for next_prod in next_products:
            keys = [(next_prod,prod) for prod in products]
            cnts = list(map(lambda k:pos_cnt_dict.get(k,0),keys))
            sort_index = np.argsort(cnts)[::-1]
            top_index = sort_index[:top] if len(cnts) > top else sort_index
            pos_items = np.array(products)[top_index]
            pos_pair[next_prod] += pos_items
    return pos_pair

def neg_pair_stats(data,pos_pair_dict):
    data = data[data['eval_set']!='test']
    prods = set(data['product_id'])
    prod_prob = data['product_id'].value_counts().to_dict()
    for key,value in prod_prob.items():
        prod_prob[key] = np.array((value / data.shape[0])**0.75).astype(np.float32)
    
    neg_pair = defaultdict(list)
    neg_pair_prob = defaultdict(list)
    items = tqdm(pos_pair_dict.items(),total=len(pos_pair_dict.keys()))
    for key,value in items:
        neg_prods = prods - set(value)
        neg_pair[key] = list(neg_prods)
        neg_prob = np.array(list(map(lambda x:prod_prob.get(x,0),neg_prods)))
        neg_prob /= neg_prob.sum()
        neg_pair_prob[key] = neg_prob.tolist()
    return neg_pair,neg_pair_prob
                
def emb_data_maker(pos_pair,neg_pair,neg_pair_prob,num_neg=5):
    samples = []
    for key,value in tqdm(pos_pair.items(),total=len(pos_pair.keys())):
        neg_samples = np.random.choice(neg_pair[key],p=neg_pair_prob[key],replace=True,size=len(value)*num_neg).reshape(-1,num_neg)
        pos_samples = np.array(value).reshape(-1,1)
        target_samples = np.array([key] * neg_samples.shape[0]).reshape(-1,1)
        sample = np.concatenate([target_samples,pos_samples,neg_samples],axis=1)
        samples.append(sample)

    samples = np.concatenate(samples)
    return samples

def emb_dataloader(prod_samples,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False
                   ):
    def batch_gen():
        total_length = prod_samples.shape[0]
        indices = np.arange(total_length)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0,total_length,batch_size):
            ind = indices[i:i+batch_size]
            if len(ind) < batch_size and drop_last:
                break
            else:
                batch = prod_samples[ind]
                yield batch
    
    for batch in batch_gen():
        batch -= 1
        batch = torch.from_numpy(batch).long().cuda()
        yield batch

get_checkpoint_path = lambda model_name,seed:f'{model_name}_best_checkpoint_{seed}.pth'
class EmbeddingTrainer:
    def __init__(self,
                 data,
                 items,
                 emb_dim,
                 epochs=10,
                 eval_epoch=1,
                 reg_lambda=0.1,
                 learning_rate=0.01,
                 batch_size=32,
                 train_size=0.8,
                 seed=97734,
                 warm_start=False,
                 early_stopping=2):
        self.data = data
        self.epochs = epochs
        self.items = items
        self.emb_dim = emb_dim
        self.train_size = train_size
        self.seed = seed
        self.eval_epoch = eval_epoch
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.loss_func = BPRLoss(reg_lambda)
    
    def train(self):
        num_items = self.data['product_id'].max()
        model = BPREmbeddingModel(num_items,self.emb_dim).cuda()
        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=0,factor=0.2,verbose=True)
        data_tr,data_val  = train_test_split(self.items,train_size=self.train_size,random_state=self.seed)
        
        model_name = model.__class__.__name__
        checkpoint_path = get_checkpoint_path(model_name,self.seed)
        checkpoint_path = f'checkpoint/{checkpoint_path}'
        if self.warm_start:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['best_epoch']
            start_epoch += 1
            best_loss = checkpoint['best_loss']
            logger.info(f'warm starting training from best epoch:{start_epoch} and best loss:{best_loss:.5f}')
        else:
            start_epoch = 0
            best_loss = np.inf
        self.checkpoint_path = checkpoint_path
        
        no_improvement = 0
        for epoch in range(start_epoch,self.epochs):
            total_loss,cur_iter = 0,1
            model.train()
            train_dl = emb_dataloader(data_tr,batch_size=self.batch_size,shuffle=True,drop_last=True)
            train_batch_loader = tqdm(train_dl,total=data_tr.shape[0]//self.batch_size,desc=f'training bpr embedding at epoch:{epoch}')
            for batch in train_batch_loader:
                optimizer.zero_grad()
                v_i,v_k,v_j = model(batch)
                loss = self.loss_func(v_i,v_k,v_j)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / cur_iter
                cur_iter += 1
                train_batch_loader.set_postfix(train_loss=f'{avg_loss:.05f}')
            
            if epoch % self.eval_epoch == 0:
                model.eval()
                total_loss_val,ite = 0,1
                eval_dl = emb_dataloader(data_val,batch_size=4096,shuffle=False,drop_last=False)
                eval_batch_loader = tqdm(eval_dl,total=data_val.shape[0]//4096,desc='evaluating bpr embedding',
                                          leave=False,dynamic_ncols=True)
                with torch.no_grad():
                    for batch in eval_batch_loader:
                        v_i,v_k,v_j = model(batch)
                        loss = self.loss_func(v_i,v_k,v_j)
                        
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
                    
                if no_improvement == self.early_stopping:
                    logger.info('early stopping is trggered,the model has stopped improving')
                    return

if __name__ == '__main__':
    path = 'data/tmp/emb_items.npy'
    data = pd.read_csv('data/orders_info.csv')
    if os.path.exists(path):
        prods = np.load(path)
    else:
        order_products,stats_cnt = pos_pair_cnt(data)
        pos_pair = pos_pair_stats(order_products,stats_cnt,top=3)
        neg_pair,neg_pair_probs = neg_pair_stats(data,pos_pair)
        prods = emb_data_maker(pos_pair,neg_pair,neg_pair_probs,num_neg=5)
        np.save(path,prods)

    for seed in [97734,876301,3985]:
        trainer =  EmbeddingTrainer(data,
                                    prods,
                                    emb_dim=24,
                                    epochs=10,
                                    eval_epoch=1,
                                    reg_lambda=0.00001,
                                    learning_rate=0.005,
                                    batch_size=4096,
                                    train_size=0.9,
                                    seed=seed,
                                    early_stopping=2)
        trainer.train()


#%%






