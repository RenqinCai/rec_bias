import os
import io
import json
import torch
import numpy as np
import random
import pandas as pd
import argparse
import pickle

from torch.utils.data import dataset 
from torch.utils.data import DataLoader

from movie import MOVIE, MOVIE_TEST
# from movie_sample import MOVIE, MOVIE_TEST
# from movie_iter import MOVIE_LOADER, MOVIE_TEST

class DATA():
    def __init__(self):
        print("data")
    
    def f_load_movie(self, args):
        self.m_data_name = args.data_name

        train_data_file = args.data_dir+"/train_data.pickle"
        valid_data_file = args.data_dir+"/valid_data.pickle"
        test_data_file = args.data_dir+"/test_data.pickle"

        train_df = pd.read_pickle(train_data_file)
        print(train_df.head())
        train_df = train_df[["userid", "pos_itemid"]]
        train_df.columns = ["userid", "itemid"]

        valid_df = pd.read_pickle(valid_data_file)
        valid_df = valid_df[["userid", "itemid"]]
        valid_df.columns = ["userid", "itemid"]

        test_df = pd.read_pickle(test_data_file)
        test_df = test_df[["userid", "itemid"]]
        test_df.columns = ["userid", "itemid"]

        self.m_vocab_file = args.vocab_file

        with open(os.path.join(args.data_dir, self.m_vocab_file), "r", encoding="utf8") as f:
            vocab = json.loads(f.read())
        
        vocab_obj = Vocab()
        vocab_obj.f_set_vocab(vocab["userindex2uid"], vocab["itemindex2iid"])
        
        train_user_num = train_df.userid.nunique()
        print("train user num", train_user_num)

        # train_item_num = train_df.itemid.nunique()
        train_item_num = train_df.itemid.nunique()
        print("train item num", train_item_num)

        train_data = MOVIE(args, train_df, vocab_obj.item_num)
        valid_data = MOVIE_TEST(args, train_df, valid_df)
        test_data = MOVIE_TEST(args, train_df, test_df)

        batch_size = args.batch_size

        # train_loader = MOVIE_LOADER(args, train_df)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate)

        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=valid_data.collate)

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=test_data.collate)

        return train_loader, valid_loader, test_loader, vocab_obj

    def f_load_tiktok(self, args):
        self.m_data_name = args.data_name

        train_data_file = args.data_dir+"/train_data.pickle"
        valid_data_file = args.data_dir+"/valid_data.pickle"
        test_data_file = args.data_dir+"/test_data.pickle"

        train_df = pd.read_pickle(train_data_file)
        train_df = train_df[["userid", "itemid", "groupid"]]
        train_df.columns = ["userid", "itemid", "groupid"]

        print("train")
        print(train_df.head())

        valid_df = pd.read_pickle(valid_data_file)
        valid_df = valid_df[["userid", "itemid", "groupid"]]
        valid_df.columns = ["userid", "itemid", "groupid"]

        print("valid")
        print(valid_df.head())

        test_df = pd.read_pickle(test_data_file)
        test_df = test_df[["userid", "itemid", "groupid"]]
        test_df.columns = ["userid", "itemid", "groupid"]

        print("test")
        print(test_df.head())

        self.m_vocab_file = args.vocab_file

        with open(os.path.join(args.data_dir, self.m_vocab_file), "r", encoding="utf8") as f:
            vocab = json.loads(f.read())
        
        vocab_obj = Vocab()
        vocab_obj.f_set_vocab(vocab["userindex2uid"], vocab["itemindex2iid"])
        
        train_user_num = train_df.userid.nunique()
        print("train user num", train_user_num)

        # train_item_num = train_df.itemid.nunique()
        train_item_num = train_df.itemid.nunique()
        print("train item num", train_item_num)

        train_data = TIKTOK(args, train_df, vocab_obj.item_num)
        valid_data = TIKTOK_TEST(args, train_df, valid_df)
        test_data = TIKTOK_TEST(args, train_df, test_df)

        batch_size = args.batch_size

        # train_loader = MOVIE_LOADER(args, train_df)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate)

        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=valid_data.collate)

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=test_data.collate)

        return train_loader, valid_loader, test_loader, vocab_obj

class Vocab():
    def __init__(self):

        self.m_user2uid = None
        self.m_item2iid = None

        self.m_user_num = 0
        self.m_item_num = 0

    def f_set_vocab(self, user2uid, item2iid):
        self.m_user2uid = user2uid
        self.m_item2iid = item2iid

        self.m_user_num = len(self.m_user2uid)
        self.m_item_num = len(self.m_item2iid)

    @property
    def user_num(self):
        return self.m_user_num
    
    @property
    def item_num(self):
        return self.m_item_num



