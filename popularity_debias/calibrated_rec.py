from datetime import datetime
import numpy as np
import torch
import random
import torch.nn as nn

import pandas as pd
import pickle
import argparse
from data import DATA
import json
import os
from optimizer import OPTIM
from logger import LOGGER

import time
from train import TRAINER
from model import MF
from eval import EVAL
from metric import get_recall

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

### data
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--data_name', type=str, default='movielens')
parser.add_argument('--data_file', type=str, default='data.pickle')

parser.add_argument('--vocab_file', type=str, default='vocab.json')
parser.add_argument('--model_file', type=str, default="model_best.pt")
parser.add_argument('--model_name', type=str, default="MF")
parser.add_argument('--model_path', type=str, default="../checkpoint/")

### model
parser.add_argument('--user_emb_size', type=int, default=300)
parser.add_argument('--item_emb_size', type=int, default=300)

parser.add_argument('--output_hidden_size', type=int, default=300)

### train
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--l2_reg', type=float, default=0.0)

parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--epoch_num', type=int, default=10)
parser.add_argument('--print_interval', type=int, default=200)
parser.add_argument('--hcdmg1', action="store_true", default=False)

### hyper-param
# parser.add_argument('--init_mult', type=float, default=1.0)
# parser.add_argument('--variance', type=float, default=0.995)
# parser.add_argument('--max_seq_length', type=int, default=100)

### others
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--parallel', action="store_true", default=False)
parser.add_argument('--local_rank', type=int, default=0)

args = parser.parse_args([])

args.data_dir = "../data/ml-1m"
args.data_name = "movie"
args.model_file = "movie_MF/model_best_2_17_10_45.pt"
args.vocab_file = "vocab.json"
args.epoch_num = 50
args.batch_size = 32
args.learning_rate = 0.0001
args.user_emb_size = 128
args.item_emb_size = 128
args.weight_decay = 0.0

ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
seed = 1234
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# local_rank = None
# if args.parallel:
#     local_rank = args.local_rank
#     torch.distributed.init_process_group(backend="nccl")
#     device = torch.device('cuda:{}'.format(local_rank))

data_obj = DATA()

if "movie" in args.data_name:
    train_data, valid_data, test_data, vocab_obj = data_obj.f_load_movie(args)

# print("vocab_size", vocab_obj.vocab_size)
print("user num", vocab_obj.user_num)
print("item num", vocab_obj.item_num)

network = MF(vocab_obj, args, device)

total_param_num = 0
for name, param in network.named_parameters():
    if param.requires_grad:
        param_num = param.numel()
        total_param_num += param_num
        print(name, "\t", param_num)

print("total parameters num", total_param_num)

print("="*10, "eval", "="*10)

eval_obj = EVAL(vocab_obj, args, device)

network = network.to(device)

def f_init_eval(network, model_file=None, reload_model=False):
    if reload_model:
        print("reload model")
        if not model_file:
            model_file = "model_best.pt"
        model_name = os.path.join(model_path, model_file)
        print("model name", model_name)
        check_point = torch.load(model_name)
        network.load_state_dict(check_point['model'])

    return network

model_path = args.model_path
network = f_init_eval(network, model_file=args.model_file, reload_model=True)

def f_eval_new(train_data, eval_data):

    recall_list = []

    print('--'*10)

    topk = 20
    network.eval()
    with torch.no_grad():

        for user_batch, item_batch, mask_item_batch, itemnum_batch in eval_data:

            user_gpu = user_batch.to(device)

            logits = network.f_eval_forward(user_gpu)

            recall = get_recall(logits.cpu(), item_batch, mask_item_batch, itemnum_batch, k=topk)

            recall_list.append(recall)

    mean_recall = np.mean(recall_list)
    print("recall@%d:%.4f"%(topk, mean_recall))

print(f_eval_new(train_data, valid_data))

train_data_file = args.data_dir+"/train_data.pickle"
valid_data_file = args.data_dir+"/valid_data.pickle"
test_data_file = args.data_dir+"/test_data.pickle"

train_df = pd.read_pickle(train_data_file)
valid_df = pd.read_pickle(valid_data_file)
test_df = pd.read_pickle(test_data_file)

valid_df = pd.concat([valid_df, test_df])

with open(os.path.join(args.data_dir, args.vocab_file), "r", encoding="utf8") as f:
    vocab = json.loads(f.read())

itemid2movieid = vocab['iid2itemindex']
itemid2movieid = {int(k):itemid2movieid[k] for k in itemid2movieid}

movieid2itemid = vocab['itemindex2iid']
movieid2itemid = {int(k):int(movieid2itemid[k]) for k in movieid2itemid if k!= "<pad>"}

title_col = "title"
genre_col = "genres"
data_dir = args.data_dir
item_info_path = os.path.join(data_dir, "movies.dat")

names = ['movieid', title_col, genre_col]
df_item = pd.read_csv(item_info_path, sep = '::', names = names)

# df_item = df_item[df_item[genre_col]!="(no genres listed)"]
# print("dimension: ", df_item.shape)
df_item.head()

class Item:
    """
    Data holder for our item.
    
    Parameters
    ----------
    id : int
  
    title : str

    genre : dict[str, float]
        The item/movie's genre distribution, where the key
        represents the genre and value corresponds to the
        ratio of that genre.

    score : float
        Score for the item, potentially generated by some
        recommendation algorithm.
    """
    def __init__(self, _id, title, genres, score=None):
        self.id = _id
        self.title = title
        self.score = score
        self.genres = genres

    def __repr__(self):
        return self.title


def create_item_mapping(df_item, item_col, title_col, genre_col):
    """Create a dictionary of item id to Item lookup."""
    item_mapping = {}
    for row in df_item.itertuples():
        item_id = getattr(row, item_col)
        item_title = getattr(row, title_col)
        item_genre = getattr(row, genre_col)

        splitted = item_genre.split('|')
        genre_ratio = 1. / len(splitted)
        item_genre = {genre: genre_ratio for genre in splitted}

        item = Item(item_id, item_title, item_genre)
        item_mapping[item_id] = item

    return item_mapping
    
item_mapping = create_item_mapping(df_item, "movieid", "title", "genres")
print(item_mapping[1])

train_df = train_df[["userid", "pos_itemid"]]

train_df.columns=["userid", "itemid"]
train_df["movieid"] = train_df.itemid.apply(lambda x: itemid2movieid[x])
train_df["movieid"] = train_df["movieid"].astype("int")

df_item['movieid'] = df_item['movieid'].astype("int")
train_df_item = train_df.merge(df_item, on="movieid")

valid_df.columns=["userid", "itemid"]
valid_df["movieid"] = valid_df.itemid.apply(lambda x: itemid2movieid[x])
valid_df["movieid"] = valid_df["movieid"].astype("int")

valid_df_item = valid_df.merge(df_item, on="movieid")

def pred4user(user_i):
    user_i = torch.tensor([user_i]).to(device)
    u = network.m_user_embed(user_i)
    ### u: batch_size*embed_size
    ### item_embed: item_size*embed_size
    # exit()
    logits = torch.matmul(u, network.m_item_embed.weight.t())
    return logits

def compute_genre_distr(items, weights=[]):
    """Compute the genre distribution for a given list of Items."""
    dist = {}
    
    item_num = len(items)
    if len(weights) == 0:
        weights = [1.0 for i in range(item_num)]
    
    assert item_num == len(weights)
    
    for idx in range(item_num):
        item = items[idx]
        weight = weights[idx]
        for genre, score in item.genres.items():
            genre_score = dist.get(genre, 0.)
            dist[genre] = genre_score + score*weight

    # we normalize the summed up probability so it sums up to 1
    # and round it to three decimal places, adding more precision
    # doesn't add much value and clutters the output
    for item, genre_score in dist.items():
        normed_genre_score = round(genre_score / len(items), 3)
        dist[item] = normed_genre_score

    return dist

topk = 20
def rec4user(user_i, mask_i, topk=20):
    mask_i = torch.from_numpy(np.array(mask_i)).to(device)
    mask_i = mask_i.unsqueeze(0)
    print(mask_i.size())
    logits = pred4user(user_i)
    
    logits.scatter_(1, mask_i, float("-inf"))
    logits[:, 0] = float("-inf")
    
    topk_val, topk_indices = torch.topk(logits, k=topk, dim=-1)
    recs = list(topk_indices.cpu()[0].numpy())
    
    return recs

topk = 20
def recpred4user(user_i, mask_i, topk=20):
    mask_i = torch.from_numpy(np.array(mask_i)).to(device)
    mask_i = mask_i.unsqueeze(0)
#     print(mask_i.size())
    logits = pred4user(user_i)
    
    logits.scatter_(1, mask_i, float("-inf"))
    logits[:, 0] = float("-inf")
    
    topk_val, topk_indices = torch.topk(logits, k=topk, dim=-1)
    recs = list(topk_indices.cpu()[0].numpy())
    scores = list(topk_val.detach().cpu()[0].numpy())
    return recs, scores

def get_recall(preds, targets, k=1):
    true_pos = set(preds)&set(targets)
    true_pos_num = len(true_pos)
    
    recall = true_pos_num/len(targets)
    
#     print("recall: %.4f"%recall)
    return recall

def compute_kl_divergence(train_distr, rec_distr, alpha=0.01):
    kl_div = 0.0
    for train_genre, train_ratio in train_distr.items():
        rec_ratio = rec_distr.get(train_genre, 0.0)
        rec_ratio = (1-alpha)*rec_ratio+alpha*train_ratio
        
        kl_div += train_ratio*np.log(train_ratio/rec_ratio)
    return kl_div

def compute_utility(rec_items, train_genre_ratio_dict, beta=0.5):
    rec_genre_ratio_dict = compute_genre_distr(rec_items)
    kl_div = compute_kl_divergence(train_genre_ratio_dict, rec_genre_ratio_dict)
    
    total_score = 0.0
    for item in rec_items:
        total_score += item.score
    
    utility = (1-beta)*total_score - beta*kl_div
    
    return utility

def calibrated_rec(items, train_genre_ratio_dict, topn, beta=0.5):
    calibrated_rec_items = []
    for _ in range(topn):
        max_utility = -np.inf
        for item in items:
            if item in calibrated_rec_items:
                continue
            
            utility = compute_utility(calibrated_rec_items+[item], train_genre_ratio_dict, beta)
            if utility > max_utility:
                max_utility = utility
                best_item = item
            
        calibrated_rec_items.append(best_item)
        
    return calibrated_rec_items

valid_user_itemlist_dict = dict(valid_df_item.groupby("userid").itemid.apply(list))
valid_unique_user_num = len(valid_user_itemlist_dict)
valid_userid_list = list(valid_user_itemlist_dict.keys())
print("user_num in valid data", valid_unique_user_num)

train_user_itemlist_dict = dict(train_df_item.groupby("userid").itemid.apply(list))
train_user_movielist_dict = dict(train_df_item.groupby("userid").movieid.apply(list))

recall_list = []
calibrated_recall_list = []

kl_list = []
calibrated_kl_list = []
# tmp_num = 2000
tmp_num = valid_unique_user_num
beta = 0.85
for u_idx in range(tmp_num):
    if u_idx % 500 == 0:
        print("u_idx", u_idx)
    valid_userid = valid_userid_list[u_idx]
    valid_target_item_list = valid_user_itemlist_dict[valid_userid]
#     valid_train_df_item = train_df_item[train_df_item.userid==valid_userid]
    
#     train_item_list = valid_train_df_item.itemid.tolist()
#     train_movieid_list = valid_train_df_item.movieid.tolist()
    
    train_item_list = train_user_itemlist_dict[valid_userid]
    train_movieid_list = train_user_movielist_dict[valid_userid]
    train_movieinfo_list = [item_mapping[i] for i in train_movieid_list]

    ## get ratio in train
    train_dist = compute_genre_distr(train_movieinfo_list)
    
    ### get calibrated rec for a user
    valid_rec_item_list, valid_rec_score_list = recpred4user(valid_userid, train_item_list, \
                                          topk=len(itemid2movieid))
    valid_rec_movie_list = [int(itemid2movieid[i]) for i in valid_rec_item_list if i!=0]
    valid_rec_movieinfo_list = []
    valid_rec_item_num = len(valid_rec_movie_list)
    for i in range(valid_rec_item_num):
        movie_i = valid_rec_movie_list[i]
        movieinfo_i = item_mapping[movie_i]
        movieinfo_i.score = valid_rec_score_list[i]
        valid_rec_movieinfo_list.append(movieinfo_i)

    ### get rec for a user
    topn = 20
    valid_top_rec_item_list = valid_rec_item_list[:topn]
    valid_top_rec_movieinfo_list = valid_rec_movieinfo_list[:topn]
    
    ### get recall for a user
    valid_recall = get_recall(valid_top_rec_item_list, valid_target_item_list, k=topn)
    recall_list.append(valid_recall)
        
    ### get genre ratio for rec
    valid_rec_dist = compute_genre_distr(valid_top_rec_movieinfo_list)
    
    ### get kl for rec
    valid_kl = compute_kl_divergence(train_dist, valid_rec_dist)
    kl_list.append(valid_kl)

    ### get calibrated rec
    calibrated_rec_movieinfo_list = calibrated_rec(valid_rec_movieinfo_list, \
                                               train_dist, topn, beta)
    calibrated_rec_dist = compute_genre_distr(calibrated_rec_movieinfo_list)
    calibrated_kl = compute_kl_divergence(train_dist, calibrated_rec_dist)
    calibrated_kl_list.append(calibrated_kl)
    
    calibrated_rec_item_list = [movieid2itemid[i.id] for i in calibrated_rec_movieinfo_list]
    calibrated_recall = get_recall(calibrated_rec_item_list, valid_target_item_list, k=topk)
    calibrated_recall_list.append(calibrated_recall)

mean_recall = np.mean(recall_list)
mean_kl = np.mean(kl_list)

mean_calibrated_recall = np.mean(calibrated_recall_list)
mean_calibrated_kl = np.mean(calibrated_kl_list)

print("recall: %.4f"%mean_recall)
print("kl: %.4f"%mean_kl)
print("calibrated recall: %.4f"%mean_calibrated_recall)
print("calibrated kl: %.4f"%mean_calibrated_kl)
