import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

def get_recall(preds, targets, mask, targetnum, k=1):
    preds = preds.view(-1, preds.size(1))

    preds.scatter_(1, mask, float("-inf"))
    preds[:, 0] = float("-inf")
    
    top_vals, indices = torch.topk(preds, k, -1)

    recall_list = []

    for i, pred_index in enumerate(indices):
        pred_i = list(pred_index.numpy())
        target_i = targets[i].numpy()
        # len_i = sum(target_i != 0)
        num_i = targetnum[i].item()
        target_i = list(target_i)[:num_i]
    
        true_pos = set(target_i) & set(pred_i)
        true_pos_num = len(true_pos)

        recall = true_pos_num/num_i
        recall_list.append(recall)

    avg_recall = np.mean(recall_list)

    return avg_recall

def get_NDCG(preds, targets, mask, targetnum, k=1):
    preds = preds.view(-1, preds.size(1))

    preds.scatter_(1, mask, float("-inf"))
    preds[:, 0] = float("-inf")
    
    top_vals, indices = torch.topk(preds, k, -1)

    ndcg_list = []

    for i, pred_index in enumerate(indices):
        dcg_i = 0
        pred_i = list(pred_index.numpy())
        target_i = targets[i].numpy()

        targetnum_i = targetnum[i].item()
        target_i = list(target_i)[:targetnum_i]

        true_pos_i = set(target_i) & set(pred_i)
        true_posnum_i = len(true_pos_i)

        for j in range(k):
            pred_ij = pred_i[j]
            
            if pred_ij in target_i:
                dcg_i += 1/np.log2(j+2)

        idcg_i = 0
        if targetnum_i > k:
            targetnum_i = k
        
        for j in range(targetnum_i):
            idcg_i += 1/np.log2(j+2)

        ndcg_i = dcg_i/idcg_i
        ndcg_list.append(ndcg_i)
        
    ndcg = np.mean(ndcg_list)

    return ndcg

# def get_pearson(preds, targets, mask, targetnum, k=1):
#     preds = preds.view(-1, preds.size(1))

#     preds.scatter_(1, mask, float("-inf"))
#     preds[:, 0] = float("-inf")
    
#     top_vals, indices = torch.topk(preds, k, -1)

#     pairwise_acc_list = []
#     for i, pred_index in enumerate(indices):
        
#         score_i = list(top_vals[i].numpy())

#         pred_i = list(pred_index.numpy())
#         target_i = targets[i].numpy()

#         targetnum_i = targetnum[i].item()
#         target_i = list(target_i)[:targetnum_i]

#         for j in range(k):
#             pred_ij = pred_i[j]
            
            
