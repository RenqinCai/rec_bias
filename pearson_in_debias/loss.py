import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

class XE_LOSS(nn.Module):
    def __init__(self, item_num, device):
        super(XE_LOSS, self).__init__()
        self.m_item_num = item_num
        self.m_device = device
    
    def forward(self, preds, targets):
        # print("==="*10)
        # print(targets.size())
        targets = F.one_hot(targets, self.m_item_num)

        # print("target", targets.size())

        # print(targets.size())
        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0

        preds = F.log_softmax(preds, 1)
        xe_loss = torch.sum(preds*targets, dim=-1)

        xe_loss = -torch.mean(xe_loss)

        # exit()

        return xe_loss

class BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(BPR_LOSS, self).__init__()
        self.m_device = device

    def forward(self, pos_preds, neg_preds, pos_mask):

        pos_sample_num = pos_preds.size(1)

        logits = []

        for i in range(pos_sample_num):
            ### logit_delta: batch_size*neg_num
            logit_delta = pos_preds[:, i].unsqueeze(1)-neg_preds
            logit_delta = logit_delta.unsqueeze(1)

            logits.append(logit_delta)

        ### logits: batch_size*pos_num*neg_num
        logits = torch.cat(logits, dim=1)

        ### loss: batch_size*pos_num*neg_num
        mask_loss = F.logsigmoid(logits)*(pos_mask.unsqueeze(-1))

        loss = torch.sum(mask_loss)

        normalizers = torch.sum(pos_mask)*neg_preds.size(1)

        loss /= normalizers

        loss = -loss

        return loss

class FAIR_LOSS(nn.Module):
    def __init__(self, device):
        super(FAIR_LOSS, self).__init__()
        self.m_device = device
    
    def forward(self, pos_preds, neg_preds, pos_mask, group):
        pos_sample_num = pos_preds.size(1)

        tau = 1e10
        epsilon = 1e-10

        pairwise_acc = None
        for i in range(pos_sample_num):
            ### logit_delta: batch_size*neg_num

            logit_delta = pos_preds[:, i].unsqueeze(1)-neg_preds
            # print("... 0 logit_delta ...", logit_delta)

            logit_delta = torch.sigmoid(logit_delta*tau+epsilon)

            # logit_delta = (logit_delta > 0).float()

            # print("... 1 logit_delta ...", logit_delta)

            pos_acc_cnt = logit_delta*(pos_mask[:, i].unsqueeze(1))
            neg_acc_cnt = -(1-logit_delta)*(pos_mask[:, i].unsqueeze(1))

            if pairwise_acc is None:
                pairwise_acc = torch.sum(pos_acc_cnt, dim=1)
                pairwise_acc += torch.sum(neg_acc_cnt, dim=1)
            else:
                pairwise_acc += torch.sum(pos_acc_cnt, dim=1)
                pairwise_acc += torch.sum(neg_acc_cnt, dim=1)

        normalizers = torch.sum(pos_mask, dim=1)*neg_preds.size(1)

        # print("pairiwse acc", pairwise_acc)
        # print("normalizers", normalizers)

        # exit()

        x = pairwise_acc / normalizers

        y = group.float()

        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        xm = x.sub(mean_x)
        ym = y.sub(mean_y)

        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den

        r_val = torch.abs(r_val)
        # print("r_val", r_val)

        return r_val