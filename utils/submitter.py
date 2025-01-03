# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:49:04 2024

@author: congx
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from itertools import product
from utils import optimize_dtypes

class F1Optimizer():

    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

def merge(data,mode='train'):
    data_tr = data[data['cate']=='train']
    data_te = data[data['cate']=='test']
    if mode == 'train':
        df_tr = data_tr[data_tr['reverse_order_number']>0]
        df_te = data_te[data_te['reverse_order_number']>1]
        df = pd.concat([df_tr,df_te])
    else:
        df = data_te[data_te['reverse_order_number']>0]
    df = df.groupby(['user_id'])['product_id'].apply(set).apply(list).reset_index()
    df = df.explode('product_id')
    files = os.listdir('metadata')
    suffix = f'{mode}.csv'
    for file in files:
        if file.endswith(suffix):
            f = pd.read_csv(f'metadata/{file}')
            # f = optimize_dtypes(f)
            cols = f.columns.tolist()
            if 'product_id' in cols and 'user_id' in cols:
                df = df.merge(f,how='left',on=['user_id','product_id'])
            elif 'product_id' in cols:
                df = df.merge(f,how='left',on=['product_id'])
            else:
                raise KeyError('product_id must be in the data')
    return df

# def merge(data,mode='train'):
#     data_tr = data[data['cate']=='train']
#     data_te = data[data['cate']=='test']
#     if mode == 'train':
#         df_tr = data_tr[data_tr['reverse_order_number']>0]
#         df_te = data_te[data_te['reverse_order_number']>1]
#         df = pd.concat([df_tr,df_te])
#     else:
#         df = data_te[data_te['reverse_order_number']>0]
#     df = df.groupby(['user_id'])['product_id'].apply(set).apply(list).reset_index()
#     df = df.explode('product_id')
#     files = [f'aff_{mode}',f'dow_tz_prob_{mode}',f'N_order_{mode}']
#     files = list(map(lambda x:x+'.csv',files))
#     for file in files:
#         f = pd.read_csv(f'metadata/{file}')
#         df = df.merge(f,how='left',on=['user_id','product_id'])
#     return df

def make_submissions(data):
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
    
    data_tr = merge(data)
    data_te = data[data['cate']=='test']
    data_te = merge(data_te,mode='test')
    
    products = pd.read_csv('data/products.csv')
    prod_aisle_dict = products.set_index('product_id')['aisle_id'].to_dict()
    columns = lambda prefix,num_cols:[f'{prefix}_{i}' for i in range(num_cols)]
    suffix = ['eval','pred']
    suffix = list(map(lambda x:x+'.npy',suffix))
    prefix = ['product','aisle','reorder']
    prefix = list(map(lambda x:'user_'+x,prefix))
    npy_files = list(map(np.load,[os.path.join('metadata',('_'.join(p))) for p in product(prefix,suffix)]))
    user_prod_eval,user_prod_pred,user_aisle_eval,user_aisle_pred,user_reorder_eval,user_reorder_pred = npy_files
    user_prod_eval[:,1] -= 1;user_prod_pred[:,1] -= 1
    label = user_prod_eval[:,-1]
    user_prod_eval  = user_prod_eval[:,:-1]
    prod_feats = columns('prod',51)
    user_prod_eval = pd.DataFrame(user_prod_eval,columns=['user_id','product_id']+prod_feats)
    user_prod_eval['aisle_id'] = user_prod_eval['product_id'].map(prod_aisle_dict)
    user_prod_pred = pd.DataFrame(user_prod_pred,columns=['user_id','product_id']+prod_feats)
    user_prod_pred['aisle_id'] = user_prod_pred['product_id'].map(prod_aisle_dict)
    user_aisle_eval = user_aisle_eval[:,:-1]
    aisle_feats = columns('aisle',51)
    user_aisle_eval = pd.DataFrame(user_aisle_eval,columns=['user_id','aisle_id']+aisle_feats)
    user_aisle_pred = pd.DataFrame(user_aisle_pred,columns=['user_id','aisle_id']+aisle_feats)
    user_reorder_eval = user_reorder_eval[:,:-1]
    reorder_feats = columns('reorder',51)
    user_reorder_eval = pd.DataFrame(user_reorder_eval,columns=['user_id']+reorder_feats)
    user_reorder_pred = pd.DataFrame(user_reorder_pred,columns=['user_id']+reorder_feats)
    nmf_emb = np.load('metadata/nmf_item_emb.npy')
    nmf_feat = columns('nmf',24)
    nmf_emb = pd.DataFrame(nmf_emb,columns=['product_id']+nmf_feat)
    user_prod_eval = user_prod_eval.merge(user_aisle_eval,how='left',on=['user_id','aisle_id']).merge(user_reorder_eval,
                                          how='left',on=['user_id']).merge(nmf_emb,how='left',on=['product_id'])

    user_prod_pred = user_prod_pred.merge(user_aisle_pred,how='left',on=['user_id','aisle_id']).merge(user_reorder_pred,
                                          how='left',on=['user_id']).merge(nmf_emb,how='left',on=['product_id'])
    user_prods = user_prod_pred[['user_id','product_id']]
    
    user_prod_eval = user_prod_eval.merge(data_tr,how='left',on=['user_id','product_id']).fillna(-1)
    del user_prod_eval['user_id'],user_prod_eval['product_id'],user_prod_eval['aisle_id']
    user_prod_eval = np.array(user_prod_eval).astype(np.float32)
    user_prod_pred = user_prod_pred.merge(data_te,how='left',on=['user_id','product_id']).fillna(-1)
    del user_prod_pred['user_id'],user_prod_pred['product_id'],user_prod_pred['aisle_id']
    user_prod_pred = np.array(user_prod_pred).astype(np.float32)
    
    dtrain = lgb.Dataset(user_prod_eval,label=label)
    model = lgb.train(params,dtrain,num_boost_round=500)
    predictions = model.predict(user_prod_pred)
    user_prods['predictions'] = predictions
    return user_prods

def submit(preds):
    predictions = np.empty([preds.shape[0],2],dtype=object)
    rows = tqdm(preds.iterrows(),total=preds.shape[0],desc='making submissions')
    for ind,row in rows:
        prod_preds_dict = dict(zip(row['product_id'], row['predictions']))
        none_prob = prod_preds_dict.get(0, None)
        del prod_preds_dict[0]
        
        other_products = np.array(list(prod_preds_dict.keys()))
        other_probs = np.array(list(prod_preds_dict.values()))
        
        idx = np.argsort(-1*other_probs)
        other_products = other_products[idx]
        other_probs = other_probs[idx]
    
        opt = F1Optimizer.maximize_expectation(other_probs, none_prob)
        best_prediction = ['None'] if opt[1] else []
        best_prediction += list(other_products[:opt[0]])
    
        predicted_products = ' '.join(map(str, best_prediction)) if best_prediction else 'None'
        predictions[ind,0] = row['order_id']
        predictions[ind,1] = predicted_products
    predictions = pd.DataFrame(predictions,columns=['order_id','products'])
    predictions.to_csv('data/submissions.csv',index=False)
    return predictions

if __name__ == '__main__':
    data = pd.read_csv('data/orders_info.csv')
    predictions = make_submissions(data)
    test_data = data[data['eval_set']=='test']
    predictions['user_id'] = predictions['user_id'].astype(np.int32)
    predictions['product_id'] = predictions['product_id'].astype(np.int32)
    user_order_dict = test_data.set_index('user_id')['order_id'].to_dict()
    predictions['order_id'] = predictions['user_id'].map(user_order_dict)
    pred_prd = predictions.groupby('order_id')['product_id'].apply(list)
    pred_pred = predictions.groupby('order_id')['predictions'].apply(list)
    preds = pd.concat([pred_prd,pred_pred],axis=1).reset_index()
    submissions = submit(preds)
    submissions.to_csv('metadata/submissions.csv',index=False)

#%%









