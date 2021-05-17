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

        batch_len = pos_preds.size(1)

        logits = []

        for i in range(batch_len):
            ### logit_delta: batch_size*neg_num
            logit_delta = pos_preds[:, i].unsqueeze(1)-neg_preds
            logit_delta = logit_delta.unsqueeze(1)

            logits.append(logit_delta)

        ### logits: batch_size*pos_num*neg_num
        logits = torch.cat(logits, dim=1)

        ### pos_mask: batch_size*pos_num
        # mask_logits = logits*(pos_mask.unsqueeze(-1))
        
        # print("logit ", logits.size())
        # print("pos_mask", pos_mask.size())

        ### loss: batch_size*pos_num*neg_num
        mask_loss = F.logsigmoid(logits)*(pos_mask.unsqueeze(-1))

        loss = torch.sum(mask_loss)

        normalizers = torch.sum(pos_mask)*neg_preds.size(1)

        loss /= normalizers

        loss = -loss

        return loss