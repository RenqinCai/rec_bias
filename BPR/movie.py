import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 

class MOVIE(Dataset):
    def __init__(self, args, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size

        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size-self.m_batch_num) > 0:
            self.m_batch_num += 1
        
        self.m_user_batch_list = []
        self.m_pos_item_batch_list = []
        self.m_neg_item_batch_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}

        userid_list = df.userid.tolist()
        pos_itemid_list = df.pos_itemid.tolist()
        neg_itemidlist_list = df.neg_itemid.tolist()

        # user_itemlist_dict = dict(df.groupby("userid").itemid.apply(list))
        # unique_userid_list = list(user_itemlist_dict.keys())
        # unique_user_num = len(unique_userid_list)

        # assert unique_user_num != df.userid.nunique()

        # for user_index in range(unique_user_num):
            # user_id = unique_userid_list[user_index]
            # itemid_list = user_itemlist_dict[user_id]
        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            pos_itemid = pos_itemid_list[sample_index]
            neg_itemid_list = neg_itemidlist_list[sample_index]

            self.m_user_batch_list.append(user_id)
            self.m_pos_item_batch_list.append(pos_itemid)
            self.m_neg_item_batch_list.append(neg_itemid_list)

        print("... load train data ...", len(self.m_user_batch_list), len(self.m_pos_item_batch_list))

    def __len__(self):
        return len(self.m_user_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i = idx

        user_i = self.m_user_batch_list[i]
        pos_item_i = self.m_pos_item_batch_list[i]
        neg_item_list_i = self.m_neg_item_batch_list[i]

        return user_i, pos_item_i, neg_item_list_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        user_iter = []
        pos_item_iter = []
        neg_item_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            user_i = sample_i[0]
            user_iter.append(user_i)

            pos_item_i = sample_i[1]
            pos_item_iter.append(pos_item_i)

            neg_item_i = sample_i[2]
            neg_item_iter.append(neg_item_i)

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        pos_item_iter_tensor = torch.from_numpy(np.array(pos_item_iter)).long()
        neg_item_iter_tensor = torch.from_numpy(np.array(neg_item_iter)).long()

        return user_iter_tensor, pos_item_iter_tensor, neg_item_iter_tensor

class MOVIE_TEST(Dataset):
    def __init__(self, args, train_df, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size

        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size-self.m_batch_num) > 0:
            self.m_batch_num += 1
        
        self.m_user_batch_list = []
        self.m_item_batch_list = []
        self.m_maskitem_batch_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()

        user_itemlist_dict = dict(df.groupby("userid").itemid.apply(list))
        unique_userid_list = list(user_itemlist_dict.keys())
        unique_user_num = len(unique_userid_list)
        print("unique_user_num", unique_user_num)
        print("user num unique in df", df.userid.nunique())

        assert unique_user_num == df.userid.nunique()

        user_maskitemlist_dict = dict(train_df.groupby("userid").pos_itemid.apply(list))

        for user_index in range(unique_user_num):
            user_id = unique_userid_list[user_index]
            itemid_list = user_itemlist_dict[user_id]
            maskitemid_list = user_maskitemlist_dict[user_id]

            self.m_user_batch_list.append(user_id)
            self.m_item_batch_list.append(itemid_list)
            self.m_maskitem_batch_list.append(maskitemid_list)

        print("... load train data ...", len(self.m_user_batch_list), len(self.m_item_batch_list))

    def __len__(self):
        return len(self.m_user_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i = idx

        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]
        maskitem_i = self.m_maskitem_batch_list[i]

        return user_i, item_i, maskitem_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        user_iter = []
        item_iter = []
        itemnum_iter = []
        max_item_num = 0

        maskitem_iter = []
        max_maskitem_num = 0

        for i in range(batch_size):
            sample_i = batch[i]
            item_i = sample_i[1]
            max_item_num = max(max_item_num, len(item_i))

            maskitem_i = sample_i[2]
            max_maskitem_num = max(max_maskitem_num, len(maskitem_i))

        pad_item_id = 0
        for i in range(batch_size):
            sample_i = batch[i]
            
            user_i = sample_i[0]
            user_iter.append(user_i)

            item_i = copy.deepcopy(sample_i[1])
            itemnum_i = len(item_i)
            itemnum_iter.append(itemnum_i)

            item_i = item_i+[pad_item_id]*(max_item_num-itemnum_i)
            item_iter.append(item_i)

            maskitem_i = copy.deepcopy(sample_i[2])
            maskitemnum_i = len(maskitem_i)
            maskitem_i = maskitem_i+[pad_item_id]*(max_maskitem_num-maskitemnum_i)
            maskitem_iter.append(maskitem_i)

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()
        
        itemnum_iter_tensor = torch.from_numpy(np.array(itemnum_iter)).long()
        maskitem_iter_tensor = torch.from_numpy(np.array(maskitem_iter)).long()

        return user_iter_tensor, item_iter_tensor, maskitem_iter_tensor, itemnum_iter_tensor