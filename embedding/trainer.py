# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:06:40 2024

@author: congx
"""
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self,
                 model,
                 user_item_nodes,
                 item_item_nodes,
                 user_neg_samples,
                 item_neg_samples,
                 loss_fn,
                 optimizer='adam',
                 epochs=1000,
                 learning_rate=0.01,
                 in_feature=100,
                 layer_sizes=[16,32,64],
                 drop_rates=[0.1,0.1,0.1],
                 random_seed=1000,
                 batch_size=32):
        self.user_item_nodes = user_item_nodes
        self.item_item_nodes = item_item_nodes
        self.user_data_idx = np.arange(self.user_item_nodes.shape[0])
        self.item_data_idx = np.arange(self.item_item_nodes.shape[0])
        
        self.user_neg_samples = user_neg_samples
        self.item_neg_samples = item_neg_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.in_feature = in_feature
        self.layer_sizes = layer_sizes
        self.drop_rates = drop_rates
        self.model = model
        self.loss_fn = loss_fn
    
    def _split(self,train_size=0.8):
        user_train_idx,user_test_idx = train_test_split(self.user_data_idx,train_size=train_size,random_state=self.random_seed)
        item_train_idx,item_test_idx = train_test_split(self.item_data_idx,train_size=train_size,random_state=self.random_seed)
        user_train,user_test = self.user_item_nodes[user_train_idx],self.user_item_nodes[user_test_idx]
        item_train,item_test = self.item_item_nodes[item_train_idx],self.item_item_nodes[item_test_idx]
        return user_train,user_test,item_train,item_test

    def _batch_generator(self,inp,drop_last=False,shuffle=True):
        indices = np.arange(inp.shape[0])
        if shuffle:
            np.random.shuffle(indices)
            
        for i in range(0,inp.shape[0],self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            if drop_last and len(batch_idx) < self.batch_size:
                break
            else:
                yield inp[batch_idx]
    
    def _get_optimizer(self):
        if self.optimizer.lower() == 'adam':
            return optim.Adam
    
    def train(self,graph,train_size=0.6,neg_user_sampling=4,neg_item_sampling=5):
        user_train,user_test,item_train,item_test = self._split(train_size=train_size)
        user_train_gen = self._batch_generator(user_train,drop_last=True,shuffle=True)
        user_test_gen = self._batch_generator(user_test,drop_last=False,shuffle=False)
        item_train = self._batch_generator(item_train,drop_last=True,shuffle=True)
        item_test = self._batch_generator(item_test,drop_last=False,shuffle=False)
        
        model = self.model(graph,
                           self.in_feature,
                           self.layer_sizes,
                           self.drop_rates)
        optimizer = self._get_optimizer()
        optimizer = optimizer(model.parameters(),lr=self.learning_rate)
        
        user_neg_dict = {idx+1:row for idx,row in enumerate(self.user_neg_samples)}
        item_neg_dict = {idx+1:row for idx,row in enumerate(self.item_neg_samples)}
        
        model.train()
        model.to('cuda')
        train_bar = tqdm(user_train_gen,total=user_train.shape[0]//self.batch_size,desc='training users')
        total_loss,batches = 0,0
        for batch in train_bar:
            batch = torch.from_numpy(batch).to('cuda')
            batch_users = batch[:,0]
            user_pos_items = batch[:,1]
            
            neg_user_samples = torch.zeros(batch.shape[0],neg_user_sampling,device='cuda')
            for idx,user in enumerate(batch_users):
                neg_samples = user_neg_dict[user.item()]
                rand_ind = torch.randperm(len(neg_samples))
                neg_samples = neg_samples[rand_ind[:neg_user_sampling]]
                neg_user_samples[idx] = torch.from_numpy(neg_samples)
                
            user_emb,pos_emb,neg_emb = model(graph,batch_users,user_pos_items,neg_user_samples)
            loss = self.loss_fn(user_emb,pos_emb,neg_emb)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batches += 1
            train_bar.set_posfix(loss=total_loss/batches)
    

#%%
