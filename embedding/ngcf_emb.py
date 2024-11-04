# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:19:39 2024

@author: congx
"""
import dgl
import torch
import numpy as np
from torch import nn
from dgl import function as fn
from torch.nn import functional as F

def construct_graph(user_item,item_item):
    total_users = user_item[:,0].max() + 1
    total_items = user_item[:,1].max() + 1
    users = [i for i in range(total_users)]
    items = [i for i in range(total_items)]
    
    link_dict = {
                  ('user','user_self','user'):(users,users),
                  ('item','item_self','item'):(items,items),
                  ('user','user_item_inte','item'):(user_item[:,0].tolist(),user_item[:,1].tolist()),
                  ('item','item_user_inte','user'):(user_item[:,1].tolist(),user_item[:,0].tolist()),
                  ('item','item_item_link','item'):(item_item[:,0].tolist(),item_item[:,1].tolist())
                    }
    num_dict = {'user':total_users,'item':total_items}
    return dgl.heterograph(link_dict,num_dict)

class Layer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 norm_dict,
                 dropout):
        super().__init__()
        self.weight1 = nn.Linear(in_feat,out_feat,bias=True)
        self.weight2 = nn.Linear(in_feat,out_feat,bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.constant_(self.weight1.bias,0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.constant_(self.weight2.bias,0)
    
    def foward(self,graph,nodes_dict):
        node_types = {}
        for srctype,etype,dsttype in graph.canonical_etypes:
            if srctype == dsttype and 'self' in etype:
                msg = self.weight1(nodes_dict[srctype])
                graph.nodes[srctype].data[etype] = msg
                node_types[(srctype,etype,dsttype)] = (
                                                        fn.copy_u(etype,'m'),
                                                        fn.sum('m','h')
                                                            )
            else:
                src,dst = graph.edges(etype=(srctype,etype,dsttype))
                norm = self.norm_dict[(srctype,etype,dsttype)]
                msg = norm * (self.weight1(nodes_dict[srctype][src]) + self.weight2(nodes_dict[srctype][src] \
                                                                         + nodes_dict[dsttype][dst]))
                graph.edges[(srctype,etype,dsttype)].data[etype] = msg
                node_types[(srctype,etype,dsttype)] = (fn.copy_e(etype,'m'),
                                                       fn.sum('m','h'))
        graph.multi_update_all(node_types,'sum')
        output = {}
        for ntype in graph.ntypes:
            h = self.leaky_relu(graph.nodes[ntype].data['h'])
            h = self.dropout(h,dim=1,p=2)
            h = F.normalize(h)
            output[ntype] = h
        return output

class NGCF(nn.Module):
    def __init__(self,
                 graph,
                 in_dim,
                 out_dim,
                 layer_sizes,
                 drop_rates):
        super().__init__()
        self.norm_dict = {}
        
        for srctype,etype,dsttype in graph.canonical_etypes:
            src,dst = graph.edges(etype=(srctype,etype,dsttype))
            dst_innodes = graph.in_degrees(dst,etype=(srctype,etype,dsttype)).float()
            src_outnodes = graph.out_degrees(src,etype=(srctype,etype,dsttype)).float()
            norm = torch.pow(src_outnodes * dst_innodes,-0.5).unsqueeze(-1)
            self.norm_dict[(srctype,etype,dsttype)] = norm
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            if i == 0:
                self.layers.append(Layer(in_dim,layer_sizes[i],self.norm_dict,drop_rates[i]))
            else:
                self.layers.append(Layer(layer_sizes[i],layer_sizes[i+1],self.norm_dict,drop_rates[i]))
        
        self.nodes_dict = {ntype:nn.Parameter(torch.nn.init.xavier_uniform_(\
                                            torch.empty(graph.num_nodes(ntype),in_dim))) for ntype in graph.ntypes}
    
    def forward(self,graph,users,pos_items,neg_items):
        pass

if __name__ == '__main__':
    user_item_interaction,item_item_interaction = data_process('data/')
    graph = construct_graph(user_item_interaction,item_item_interaction)

#%%
# src,dst = graph.edges(etype=('item','item_item_link','item'))
for s,e,t in graph.canonical_etypes:
    print(s,e,t)
    


#%%
# x = torch.randn(32,1,10)
x.squeeze()



