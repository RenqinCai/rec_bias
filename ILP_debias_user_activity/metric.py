import numpy as np
import pandas as pd


def get_ndcg(preds, targets, topk=20):
    
    target_num = len(targets)
    k = min(target_num, topk)

    dcg = 0
    for i in range(topk):
        pred_i = preds[i]
        if pred_i in targets:
            dcg += 1/np.log2(i+2)

    idcg = 0
    for i in range(k):
        idcg += 1/np.log2(i+2)

    ndcg = dcg/idcg

    return ndcg


def get_recall(preds, targets, topk=20):

    target_num = len(targets)
    true_pos = set(preds) & set(targets)

    true_pos_num = len(true_pos)
    recall = true_pos_num*1.0/target_num

    return recall


def get_precision(preds, targets, topk=20):
    true_pos = set(preds) & set(targets)
    true_pos_num = len(true_pos)

    precision = true_pos_num*1.0/topk

    return precision

def get_F1(preds, targets, topk=20):
    target_num = len(targets)
    true_pos = set(preds) & set(targets)

    true_pos_num = len(true_pos)
    recall = true_pos_num*1.0/target_num

    precision = true_pos_num*1.0/topk

    F1 = recall*precision*2/(recall+precision)

    return F1