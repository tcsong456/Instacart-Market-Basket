# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:58:07 2024

@author: congx
"""
import dgl
import torch
import itertools
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import dgl.function as fn
from utils import load_data,get_label
from torch.utils.data import Dataset,DataLoader
from torch import nn
from pandas.api.types import is_float_dtype
from collections import defaultdict
from utils import optimize_dtypes

class GlobalGatedUpdate(nn.Module):
    def __init__(self,embedding_matrix):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.alpha = nn.Parameter(torch.rand(49688,1)).to('cuda')
    
    def forward(self,
                nodes,
                x):
        start = 0
        batch_embedding = []
        emb_matrix = self.embedding_matrix(torch.tensor([i for i in range(49688)]).to('cuda'))
        for node in nodes:
            num_nodes = node.shape[0]
            node_features = x[start:start+num_nodes]
            beta = torch.zeros(49688,1).to('cuda')
            beta[node] = 1
            embed = (1 - beta * self.alpha) * emb_matrix.clone()
            embed[node,:] = embed[node,:] + self.alpha[node] * node_features
            batch_embedding.append(embed)
        batch_embedding = torch.stack(batch_embedding)
        return batch_embedding
            

class AggregatedTemporalInfo(nn.Module):
    def __init__(self,
                 input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,1)
    
    def forward(self,xs):
        agg_features = []
        for x in xs:
            weights = self.linear(x).transpose(1,2)
            agg_feature = weights.matmul(x).squeeze(dim=1)
            agg_features.append(agg_feature)
        agg_features = torch.cat(agg_features,dim=0)
        return agg_features

class SelfAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        
        assert output_dim % n_heads == 0,'output dimetion must be multiple times the number of heads used'
        self.per_head_dim = self.dq = self.dk = self.dv = self.output_dim // self.n_heads
        
        self.wq = nn.Linear(input_dim, output_dim)
        self.wk = nn.Linear(input_dim, output_dim)
        self.wv = nn.Linear(input_dim, output_dim)
    
    def forward(self,x):
        n_nodes,seq_length = x.shape[0],x.shape[1]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.reshape(n_nodes,seq_length,self.n_heads,self.dq).transpose(1,2)
        k = k.reshape(n_nodes,seq_length,self.n_heads,self.dk).permute(0,2,3,1)
        v = v.reshape(n_nodes,seq_length,self.n_heads,self.dv).transpose(1,2)
        
        attention_score = q.matmul(k) / np.sqrt(self.per_head_dim)
        attention_mask = torch.zeros(seq_length,seq_length).masked_fill(torch.tril(torch.ones(seq_length,seq_length)) == 0,-np.inf).to('cuda')
        attention_score += attention_mask
        attention_score = torch.softmax(attention_score, dim=-1)
        mul_head = attention_score.matmul(v)
        mul_head = mul_head.reshape(n_nodes,seq_length,self.n_heads*self.dv)
        return mul_head

class WeightedGraphConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features,out_features,bias=True)
    
    def forward(self,graph,node_features,edge_weights):
        graph = graph.local_var()
        graph = graph.to('cuda')
        edge_weights = edge_weights.to('cuda')
        graph.ndata['n'] = node_features
        graph.edata['e'] = edge_weights.t().unsqueeze(-1)
        graph.update_all(fn.u_mul_e('n','e','msg'),fn.sum('msg','h'))
        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        return output

class WeightedGCN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_sizes,
                 out_features):
        super().__init__()
        gcns,relus,bns = nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        for hidden_size in hidden_sizes:
            gcns.append(WeightedGraphConv(in_features,hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            in_features = hidden_size
        
        gcns.append(WeightedGraphConv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(hidden_size))
        self.gcns,self.relus,self.bns = gcns,relus,bns
    
    def forward(self,graph,node_features,edge_weights):
        h = node_features
        for gcn,relu,bn in zip(self.gcns,self.relus,self.bns):
            h = gcn(graph,h,edge_weights)
            h = bn(h.transpose(1,-1)).transpose(1,-1)
            h = relu(h)
        return h
    
class DNNSTP(nn.Module):
    def __init__(self,
                 n_items,
                 emb_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items,emb_dim)
        self.n_items = n_items
        self.gcn = WeightedGCN(emb_dim,[emb_dim],emb_dim)
        self.masked_self_att = SelfAttention(emb_dim, emb_dim)
        self.agg_feature = AggregatedTemporalInfo(emb_dim)
        self.gated_update = GlobalGatedUpdate(self.item_embedding)
        self.final = nn.Linear(emb_dim,1)
    
    def forward(
        self,
        batch_graph,
        batch_nodes_feature,
        batch_edges_weight,
        batch_lengths,
        batch_nodes,
    ):

        batch_nodes_output = [self.gcn(graph,nodes,edges_weight) for graph,nodes,edges_weight in zip( \
                              batch_graph,batch_nodes_feature,batch_edges_weight)]
        batch_nodes_output = [self.masked_self_att(batch_nodes) for batch_nodes in batch_nodes_output]
        batch_nodes_output = self.agg_feature(batch_nodes_output)
        batch_nodes_output = self.gated_update(batch_nodes,batch_nodes_output)
        final_score = self.final(batch_nodes_output).squeeze(-1)
            
        return final_score
        

class DnnstpSet(Dataset):
    def __init__(self,
                 data_path,
                 item_embedding):
        
        
        
        
        self.user_products = user_products.to_dict()
        self.item_embedding = item_embedding
    
    def __getitem__(self,index):
        user_baskets = self.user_products[index+1]
        
        unique_items = torch.unique(torch.tensor(list(itertools.chain.from_iterable(user_baskets[:-1]))))
        products_embedding = self.item_embedding(unique_items.to("cuda"))
        nodes = torch.tensor(list(range(products_embedding.shape[0])))
        src = np.stack([nodes for _ in range(nodes.shape[0])],axis=1).flatten().tolist()
        dst = np.stack([nodes for _ in range(nodes.shape[0])],axis=0).flatten().tolist()
        
        g = dgl.graph((src,dst),num_nodes=nodes.shape[0])
        edge_weight_dict = defaultdict(float)
        for basket in user_baskets:
            for i in range(len(basket)):
                itemi = basket[i]
                for j in range(i+1,len(basket)):
                    itemj = basket[j]
                    edge_weight_dict[(itemi,itemj)] += 1.0
                    edge_weight_dict[(itemj,itemi)] += 1.0

        for item in unique_items.tolist():
            if edge_weight_dict[(item,item)] == 0.0:
                edge_weight_dict[(item,item)] = 1.0
        max_weight = max(edge_weight_dict.values())
        for key,value in edge_weight_dict.items():
            edge_weight_dict[key] = value / max_weight

        edge_weights = []
        for basket in user_baskets[:-1]:
            edge_weight = []
            for item_1 in unique_items.tolist():
                for item_2 in unique_items.tolist():
                    if (item_1 in basket and item_2 in basket) or (item_1 == item_2):
                        edge_weight.append(edge_weight_dict[(item_1,item_2)])
                    else:
                        edge_weight.append(0.0)
            edge_weights.append(torch.Tensor(edge_weight))
        edge_weights = torch.stack(edge_weights)
        
        return g,products_embedding,edge_weights,unique_items,user_baskets
    
    def __len__(self):
        return len(self.user_products)

# def collate(batch):
#     ret = []
#     for idx,data in enumerate(zip(*batch)):
#         if isinstance(data[0],dgl.DGLGraph):
#             ret.append(dgl.batch(data))
#         elif isinstance(data[0],torch.Tensor):
#             if idx == 2:
#                 max_length = max([d.shape[0] for d in data])
#                 edge_weights,data_lengths = [],[]
#                 for d in data:
#                     if d.shape[0] < max_length:
#                         edge_weights.append(torch.cat((d,torch.stack([torch.eye(int(d.shape[1]**0.5)).flatten() \
#                                               for _ in range(max_length - d.shape[0])],dim=0)),dim=0))
#                     else:
#                         edge_weights.append(d)
#                     data_lengths.append(d.shape[0])
#                 ret.append(torch.cat(edge_weights,dim=1))
#                 ret.append(torch.tensor(data_lengths))
#             else:
#                 ret.append(torch.cat(data,dim=0))
#         elif isinstance(data[0],list):
#             user_data_batch = data
#         else:
#             raise ValueError('unknown data type {} is found'.format(type(data)))
    
#     y = get_label([user_data[-1] for user_data in user_data_batch],num_classes=49688)
#     ret.append(y)
    
#     user_freq = torch.zeros([len(batch),49688])
#     for idx,user_baskets in enumerate(user_data_batch):
#         for user_basket in user_baskets:
#             for item in user_basket:
#                 user_freq[idx,item] += 1
#     ret.append(torch.Tensor(user_freq))
#     return ret

def collate(batch):
    ret = []
    graphs,lengths = [],[]
    for data in zip(*batch):
        if isinstance(data[0],dgl.DGLGraph):
            for g in data:
                graphs.append(g)
            ret.append(graphs)
        elif isinstance(data[0],list):
            for d in data:
                lengths.append(len(d)-1)
            ret.append(lengths)
            y = get_label([d[-1] for d in data],num_classes=49688)
            ret.append(y)
        else:
            ret.append(data)

    return ret
            
    

if __name__ == '__main__':
    # data_dict = load_data('data/')
    # orders = data_dict['orders']
    # orders_prior,orders_train = data_dict['order_products__prior'],data_dict['order_products__train']
    # products = data_dict['products']
    # aisles,departments = data_dict['aisles'],data_dict['departments']
    
    # order_products = pd.concat([orders_prior,orders_train])
    # del orders_prior,orders_train 
    # orders = orders.merge(order_products,how='left',on='order_id').merge(products,\
    #           how='left',on='product_id').merge(aisles,how='left',on='aisle_id').merge(departments,how='left',on='department_id')
    # orders = orders[orders['eval_set']!='test']
    # del order_products
    # for col,dtype in zip(orders.dtypes.index,orders.dtypes.values):
    #     if is_float_dtype(dtype):
    #         if pd.isnull(orders[col]).any():
    #             orders[col].fillna(0,inplace=True)
    #         orders[col] = orders[col].astype(np.int64)
    # orders = optimize_dtypes(orders)
    # user_order_products = orders.groupby(['user_id','order_id','order_number'])['product_id'].apply(list)
    # user_order_products = user_order_products.reset_index().sort_values(['user_id','order_number'])
    # user_products = user_order_products.groupby(['user_id'])['product_id'].apply(list)
    
    model = DNNSTP(49688,100).to('cuda')
    dataset = DnnstpSet('data/',model.item_embedding)
    dl = DataLoader(dataset=dataset,
                    batch_size=8,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=collate)
    cnt = 0
    for batch in dl:
        cnt += 1
        graph,nodes_features,edge_weights,nodes,lengths,y = batch
        if cnt == 1:
            results = model(graph,nodes_features,edge_weights,lengths,nodes)
            break
                


#%%
s = 0
for b in batch[3]:
    s += b.shape[0]
    print(b.shape[0])
s

#%%
# import time
# start = time.time()
# model = DNNSTP(49688,100)
# ds = DnnstpSet('data/',model.item_embedding)
# end = time() - start
batch[-3][0].shape

#%%
results.shape









