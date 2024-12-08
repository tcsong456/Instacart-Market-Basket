# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:05:23 2024

@author: congx
"""
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = pd.read_csv('data/orders_info.csv')
data = data[data['eval_set']!='test']
data['counter'] = 1
nmf = NMF(n_components=30,max_iter=200)

if __name__ == '__main__':
    user_item_cnt = np.array(data.groupby(['user_id','product_id'])['counter'].sum().reset_index()).astype(np.int32)
    users = data['user_id'].max()
    items = data['product_id'].max()
    interaction_matrix = np.zeros([users,items],dtype=np.int32)
    for u,i,c in user_item_cnt:
        u -= 1;i -= 1
        interaction_matrix[u,i] = c
    interaction_matrix = sparse.csr_matrix(interaction_matrix)
    
    user_emb = nmf.fit_transform(interaction_matrix)
    item_emb = nmf.components_
    user_ids = np.array([i for i in range(1,users+1)]).reshape(-1,1)
    item_ids = np.array([i for i in range(1,items+1)]).reshape(-1,1)
    user_emb = np.concatenate([user_ids,user_emb],axis=1).astype(np.float32)
    item_emb = np.concatenate([item_ids,item_emb.T],axis=1).astype(np.float32)
    np.save('metadata/nmf_user_emb.npy',user_emb)
    np.save('metadata/nmf_item_emb.npy',item_emb)
    
#%%
# item_embedding = TSNE(n_components=2,random_state=3398).fit_transform(item_emb.T)
# plt.scatter(item_embedding[:,0],item_embedding[:,1])
# plt.title('Item_embedding(t-SNE)')
# plt.xlabel('Dimention1')
# plt.ylabel('Dimention2')
# np.linalg.norm(np.dot(user_emb,item_emb)


#%%
