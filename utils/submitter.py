# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:49:04 2024

@author: congx
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    
def make_submissions(preds):
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
    test_data = data[data['eval_set']=='test']

    predictions = pd.read_csv('metadata/user_product_prob.csv')
    predictions['user_id'] = predictions['user_id'].astype(np.int32)
    predictions['product_id'] = predictions['product_id'].astype(np.int32)
    user_order_dict = test_data.set_index('user_id')['order_id'].to_dict()
    predictions['order_id'] = predictions['user_id'].map(user_order_dict)
    pred_prd = predictions.groupby('order_id')['product_id'].apply(list)
    pred_pred = predictions.groupby('order_id')['predictions'].apply(list)
    preds = pd.concat([pred_prd,pred_pred],axis=1).reset_index()
    submissions = make_submissions(preds)
    submissions.to_csv('metadata/submissions.csv',index=False)

#%%
