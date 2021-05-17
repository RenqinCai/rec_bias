import torch
from torch import log
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class MF(torch.nn.Module):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device

        n_users = vocab_obj.user_num
        n_items = vocab_obj.item_num
        n_user_embed_size = args.user_emb_size
        n_item_embed_size = args.item_emb_size

        # self.m_user_embed = nn.Parameter(torch.empty(n_users, n_user_embed_size))
        # self.m_item_embed = nn.Parameter(torch.empty(n_items, n_item_embed_size))

        self.m_user_embed = nn.Embedding(n_users, n_user_embed_size)
        self.m_item_embed = nn.Embedding(n_items, n_item_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        # nn.init.normal_(self.m_user_embed.weight, std=0.01)
        # nn.init.normal_(self.m_item_embed.weight, std=0.01)
        initrange = 0.1
        torch.nn.init.uniform_(self.m_user_embed.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embed.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()

        mask = torch.arange(0, max_len).expand(len(length), max_len).to(self.m_device)
        mask = mask < length.unsqueeze(1)

        return mask

    def forward(self, user, pos_item, neg_item, pos_itemnum):
        ## user_embed: batch_size*embed_size
        # user_embed = self.m_user_embed(user)

        # u = self.m_user_embed[user, :]
        u = self.m_user_embed(user)
        pos_i = self.m_item_embed(pos_item)
        neg_j = self.m_item_embed(neg_item)

        ### u: batch_size*embed_size
        ### pos_i: batch_size*embed_size
        ### neg_j: batch_size*neg_num*embed_size

        ### logit_ui: batch_size*pos_num
        logit_ui = (u.unsqueeze(1)*pos_i).sum(dim=-1)
        # print("logit_ui", logit_ui.size())

        ### logit_uj: batch_size*neg_num
        logit_uj = (u.unsqueeze(1)*neg_j).sum(dim=-1)

        # logits = torch.cat([logit_ui, logit_uj], dim=-1)
        
        pos_mask = self.f_generate_mask(pos_itemnum)
        
        return logit_ui, logit_uj, pos_mask

    def f_eval_forward(self, user):

        # print(user)
        # u = self.m_user_embed[user, :]
        u = self.m_user_embed(user)
        ### u: batch_size*embed_size
        ### item_embed: item_size*embed_size
        logits = torch.matmul(u, self.m_item_embed.weight.t())

        return logits