import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import dgl
import torch
import multiprocessing
import pickle
import os

import random
from time import time
from collections import defaultdict
import warnings
import scipy, gc

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)

def load_data_both(model_args):
    global args
    args = model_args

    output_path = args.data_path + '/DianPing' + '/graph_tag_1002.pkl'
    with open(output_path, 'rb') as f:
        graph_dp = pickle.load(f)
        graph_dp_tag = pickle.load(f)
    print('load graph')
    del graph_dp
    gc.collect()

    output_path = args.data_path + '/DianPing' + '/train_test_1002.pkl'
    with open(output_path, 'rb') as f:
        train_cf = pickle.load(f)
        test_cf = pickle.load(f)
        len_item = pickle.load(f)
        train_user_set = pickle.load(f)
        test_user_set = pickle.load(f)
    print('load train_test')

    output_path = args.data_path + '/DianPing' + '/str2id_1002.pkl'
    with open(output_path, 'rb') as f:
        user2idx = pickle.load(f)
        review2idx = pickle.load(f)
        POI2idx = pickle.load(f)
        query2idx = pickle.load(f)
        tag2idx = pickle.load(f)
    print('load str_id')
    tag2idx_inv = dict(zip(list(tag2idx.values()), list(tag2idx.keys())))

    # user group
    threshold = 5
    cs_user = []
    inactive_user = []
    active_user = []
    for u_id in tqdm(train_user_set):
        if len(train_user_set[u_id]) > 0 and len(train_user_set[u_id]) < threshold:
            inactive_user.append(u_id)
        else:
            active_user.append(u_id)
    u_ids = list(test_user_set.keys())
    cs_user = list(set(list(test_user_set.keys())) - set(list(train_user_set.keys())))

    # print all parameters
    global n_users, n_items
    n_users = graph_dp_tag.num_nodes(ntype='user')
    n_items = graph_dp_tag.num_nodes(ntype='review')
    n_item4rs = len_item
    n_tag = graph_dp_tag.num_nodes(ntype='tag')

    n_urt = graph_dp_tag.num_edges(etype='u_r_t')
    n_uqt = graph_dp_tag.num_edges(etype='u_q_t')
    n_uprt = graph_dp_tag.num_edges(etype='u_p_r_t')
    n_rht = graph_dp_tag.num_edges(etype='r_h_t')

    print('cold start user: ', len(cs_user), len(cs_user) / n_users)
    print('inactive user: ', len(inactive_user), len(inactive_user) / n_users)
    print('active user: ', len(active_user), len(active_user) / n_users)

    print('user: ', n_users)
    print('item: ', n_item4rs)
    print('review: ', n_items)
    print('item/review: ', n_item4rs / n_items * 100)

    print('intra-domain average tag: ', n_urt / n_users)
    print('search-based cross-domain average tag: ', n_uqt / n_users)
    print('consume-based cross-domain average tag: ', n_uprt / n_users)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_items4rs': int(n_item4rs),
        'n_tag': int(n_tag)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph_dp_tag

def load_data_both_amazon(model_args):
    global args
    args = model_args

    directory = args.data_path + args.dataset + '/'

    # load graph
    output_path = args.data_path + '/DianPing' + '/graph_tag_book2movie2.pkl'
    with open(output_path, 'rb') as f:
        graph_dp = pickle.load(f)
        graph_dp_tag = pickle.load(f)
    print('load graph')
    # del graph_dp
    gc.collect()

    output_path = args.data_path + '/DianPing' + '/train_test_book2movie2.pkl'
    with open(output_path, 'rb') as f:
        train_cf = pickle.load(f)
        test_cf = pickle.load(f)
        len_item = pickle.load(f)
        train_user_set = pickle.load(f)
        test_user_set = pickle.load(f)
    print('load train_test')

    # user group
    threshold = 5
    cs_user = []
    inactive_user = []
    active_user = []
    for u_id in tqdm(train_user_set):
        if len(train_user_set[u_id]) > 0 and len(train_user_set[u_id]) < threshold:
            inactive_user.append(u_id)
        else:
            active_user.append(u_id)
    u_ids = list(test_user_set.keys())
    cs_user = list(set(list(test_user_set.keys())) - set(list(train_user_set.keys())))

    a = 1
    # print all parameters
    global n_users, n_items
    n_users = graph_dp_tag.num_nodes(ntype='user')
    n_items = graph_dp_tag.num_nodes(ntype='review')
    n_item4rs = len_item
    n_tag = graph_dp_tag.num_nodes(ntype='tag')

    n_urt = graph_dp_tag.num_edges(etype='u_i_t_t')
    n_uprt = graph_dp_tag.num_edges(etype='u_i_s_t')
    n_rht = graph_dp_tag.num_edges(etype='r_h_t')

    print('cold start user: ', len(cs_user), len(cs_user) / n_users)
    print('inactive user: ', len(inactive_user), len(inactive_user) / n_users)
    print('active user: ', len(active_user), len(active_user) / n_users)

    print('user: ', n_users)
    print('item: ', n_item4rs)
    print('review: ', n_items)
    print('item/review: ', n_item4rs / n_items * 100)

    print('intra-domain average tag: ', n_urt / n_users)
    print('cross-domain average tag: ', n_uprt / n_users)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_items4rs': int(n_item4rs),
        'n_tag': int(n_tag)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph_dp_tag