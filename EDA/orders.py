# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:06:28 2024

@author: congx
"""
import matplotlib as mpl
mpl.set_loglevel('warning')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data,logger,split_time_zone

data = load_data('data/')
keys = list(data.keys())
for d_name,d_data in data.items():
    if d_name not in globals():
        globals()[d_name] = d_data
del data,d_name,d_data
 
def basic_info_orders():
    unique_users_train = set(orders[orders['eval_set']=='train']['user_id'])
    unique_users_test = set(orders[orders['eval_set']=='test']['user_id'])
    common_users = list(unique_users_train & unique_users_test)
    logger.debug('number of common users for train and test set is {}'.format(len(common_users)))
    
    num_users_train,num_users_test = len(unique_users_train),len(unique_users_test)
    plt.figure(figsize=(4,4))
    plt.bar(['users_train','users_test'], [num_users_train,num_users_test],color=['tab:red','tab:blue'])
    plt.ylabel('number of users')
    plt.title('Train Test Users')
    
    fig,axs = plt.subplots(2,2,figsize=(6,4),layout='constrained')
    dow_cnt = np.array(orders.value_counts(orders['order_dow']))
    sorted_indexes = np.argsort(orders.value_counts(orders['order_dow']).index.tolist())
    dow_cnt = dow_cnt[sorted_indexes]
    axs[0,0].bar(['Mon','Tue','Wed','Th','Fri','Sat','Sun'],
                 dow_cnt)

    hour_cnt = orders.value_counts(orders['order_hour_of_day']).head(10)
    hour = hour_cnt.index.tolist()
    axs[0,1].bar(range(len(hour)),list(hour_cnt),color=['tab:red'])
    axs[0,1].set_xticks(range(len(hour)))
    axs[0,1].set_xticklabels(hour)

    orders_no_prior = orders[orders['days_since_prior_order'] > 0]
    since_prior_order_cnt = orders_no_prior.value_counts(orders['days_since_prior_order'].astype(np.int8)).head(10)
    since_prior_order = since_prior_order_cnt.index.tolist()
    axs[1,0].bar(range(len(since_prior_order)),since_prior_order_cnt,color=['tab:green'])
    axs[1,0].set_xticks(range(len(since_prior_order)))
    axs[1,0].set_xticklabels(since_prior_order)
    del orders_no_prior

    dow_mapping = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    orders['order_dow'] = orders['order_dow'].map(dow_mapping)
    orders['time_zone'] = orders['order_hour_of_day'].apply(split_time_zone)
    orders['dow_timezone'] = orders['order_dow'] + '_' + orders['time_zone']
    dow_timezone_cnt = orders.value_counts(orders['dow_timezone']).head(10)
    dow_timezone = dow_timezone_cnt.index.tolist()
    axs[1,1].bar(range(len(dow_timezone)),since_prior_order_cnt,color=['tab:orange'])
    axs[1,1].set_xticks(range(len(dow_timezone)))
    axs[1,1].set_xticklabels(dow_timezone,rotation=90)
    
    plt.figure(figsize=(2,2))
    order_freq = orders.groupby('user_id')['order_number'].max().tolist()
    plt.hist(order_freq,bins=30)
    plt.ylabel('number of times',fontsize=8)
    plt.xlabel('number of orders per user',fontsize=8)
    plt.xticks([0,10,25,50,75,100])

#histogram of reorder rate for users
order_products = pd.concat([order_products__prior,order_products__train])
del order_products__prior,order_products__train
orders = orders.merge(order_products,how='left',on='order_id')
user_cnt = orders.groupby('user_id')['user_id'].count()
user_reorder_cnt = orders.groupby('user_id')['reordered'].sum()
user_reoreder_rate = user_reorder_cnt / user_cnt
plt.hist(user_reoreder_rate,bins=50)
plt.xlabel('user_reorder_rate',fontsize=14)

product_reorder_cnt = orders.groupby('product_id')['reordered'].sum()
product_cnt = orders.groupby('product_id')['reordered'].count()
product_reorder_rate = product_reorder_cnt / product_cnt
product_reorder_rate_ = product_reorder_rate[product_reorder_rate > 0]
plt.hist(product_reorder_rate_,bins=40)
plt.title('Historgram of reorder rate by user and product')
plt.legend()

fig,ax = plt.subplots()
top_reordered_product = product_reorder_rate.sort_values(ascending=False).head(15)
reorderd_products = list(map(str,(map(int,top_reordered_product.index.tolist()))))
ax.bar(range(len(reorderd_products)),top_reordered_product.tolist())
ax.set_xticks(range(len(reorderd_products)))
ax.set_xticklabels(reorderd_products,rotation=45)
ax.set_ylim(0.5)
ax.set_title('top 15 products with highest reorder rate')
    
if __name__ == '__main__':
    basic_info_orders()

#%%






        



