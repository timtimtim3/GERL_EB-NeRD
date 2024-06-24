# -*- coding: utf-8 -*-
"""
构建两个1-hop关系表
- news index => user index list, 可以提前做好sample
- user index => news index list, 可以提前做好sample

构建两个 2-hop关系表
- news index => 2-hop news index list, 可以提前做好sample
- user index => 2-hop user index list, 可以提前做好sample

只能利用 hist 信息构建这个表，只保留在train中出现过的 user 和 item
train 和 val的时候共用上边的表，都是基于train的hist来构建
test 单独跑，基于trian + test 的hist来构建
"""
from collections import defaultdict
import os
import json
import random
import argparse
import sys
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.vocab import WordVocab

ROOT_PATH = os.environ["GERL"]

def build_two_hop_neighbors(cfg, user_one_hop: Dict, news_one_hop: Dict, part: str):
    user_dict = dict()
    news_dict = dict()
    for user, news_list in tqdm(user_one_hop.items(), desc="Building hop-2 user"):
        two_hop_users = []
        for news in news_list:
            two_hop_users += news_one_hop[news]
        if len(two_hop_users) > cfg.max_user_two_hop:
            two_hop_users = random.sample(two_hop_users, cfg.max_user_two_hop)
        user_dict[user] = two_hop_users
    for news, user_list in tqdm(news_one_hop.items(), desc="Building hop-2 news"):
        two_hop_news = []
        for user in user_list:
            two_hop_news += user_one_hop[user]
        if len(two_hop_news) > cfg.max_news_two_hop:
            two_hop_news = random.sample(two_hop_news, cfg.max_news_two_hop)
        news_dict[news] = two_hop_news
    
    f_user = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "{}-user_two_hops.txt".format(part))
    f_news = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "{}-news_two_hops.txt".format(part))
    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list])
            fw.write("{}\t{}\n".format(news, user_list_str))


def build_two_hop_neighbors_ranked(cfg, user_one_hop: Dict, news_one_hop: Dict, part: str):
    user_dict = defaultdict(list)
    news_dict = defaultdict(list)

    # Convert user_one_hop lists to sets for faster intersection computation
    user_click_sets = {user: set(news_list) for user, news_list in user_one_hop.items()}

    # Compute user-user common news click counts using set intersections
    user_common_clicks = defaultdict(lambda: defaultdict(int))
    for user, user_clicks in tqdm(user_click_sets.items(), desc="Building user common click counts"):
        for other_user, other_clicks in user_click_sets.items():
            if user != other_user:
                common_clicks_count = len(user_clicks.intersection(other_clicks))
                if common_clicks_count > 0:
                    user_common_clicks[user][other_user] = common_clicks_count

    # Build two-hop user neighbors based on common clicks
    for user, common_users in tqdm(user_common_clicks.items(), desc="Building hop-2 user"):
        # Sort the users based on common click counts in descending order
        sorted_users = sorted(common_users.items(), key=lambda x: x[1], reverse=True)
        top_users = [other_user for other_user, count in sorted_users[:cfg.max_user_two_hop]]
        user_dict[user] = top_users

    # Build two-hop news neighbors based on user one-hop neighbors
    for news, user_list in tqdm(news_one_hop.items(), desc="Building hop-2 news"):
        two_hop_news = []
        for user in user_list:
            two_hop_news += user_one_hop[user]
        if len(two_hop_news) > cfg.max_news_two_hop:
            two_hop_news = random.sample(two_hop_news, cfg.max_news_two_hop)
        news_dict[news] = two_hop_news

    # Determine file suffix based on flags
    suffix = ""
    if cfg.sort_one_hop_by_read_time and cfg.rank_two_hop_by_common_clicks:
        suffix = "_rt_ranked"
    elif cfg.rank_two_hop_by_common_clicks:
        suffix = "_ranked"
    elif cfg.sort_one_hop_by_read_time:
        suffix = "_rt"

    f_user = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"{part}-user_two_hops{suffix}.txt")
    f_news = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"{part}-news_two_hops{suffix}.txt")

    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join(map(str, news_list))
            fw.write(f"{user}\t{news_list_str}\n")

    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join(map(str, user_list))
            fw.write(f"{news}\t{user_list_str}\n")


def build_one_hop_neighbors(cfg, behavior_df: pd.DataFrame, user_vocab: WordVocab, newsid_vocab: WordVocab, part: str) -> Tuple[Dict, Dict]:
    # behavior_df = behavior_df.fillna("")
    user_dict = dict()
    news_dict = dict()

    for uid, hist in tqdm(behavior_df[["uid", "hist"]].values, desc="Building Hop-1"):
        if uid not in user_vocab.stoi:
            continue
        user_index = user_vocab.stoi[uid]

        if user_index not in user_dict:
            user_dict[user_index] = []

        for newsid in hist:
            newsid = str(newsid)
            if newsid not in newsid_vocab.stoi:
                continue
            news_index = newsid_vocab.stoi[newsid]
            if news_index not in news_dict:
                news_dict[news_index] = []
            # click_list.append([user_index, news_index])
            if len(user_dict[user_index]) < cfg.max_user_one_hop:
                user_dict[user_index].append(news_index)
            if len(news_dict[news_index]) < cfg.max_news_one_hop:
                news_dict[news_index].append(user_index)
    
    f_user = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "{}-user_one_hops.txt".format(part))
    f_news = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "{}-news_one_hops.txt".format(part))
    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list[:cfg.max_user_one_hop]])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list[:cfg.max_news_one_hop]])
            fw.write("{}\t{}\n".format(news, user_list_str))

    return user_dict, news_dict


def build_one_hop_neighbors_rt(cfg, behavior_df: pd.DataFrame, user_vocab: WordVocab, newsid_vocab: WordVocab, part: str) -> Tuple[Dict, Dict]:
    user_dict = {}
    news_dict = {}
    news_dict_rt = {}

    for uid, hist, read_time in tqdm(behavior_df[["uid", "hist", "read_time"]].values, desc="Building Hop-1"):
        if uid not in user_vocab.stoi:
            continue
        user_index = user_vocab.stoi[uid]

        if user_index not in user_dict:
            user_dict[user_index] = []

        # Sort read_time and hist such that the largest read_time comes first
        sorted_pairs = sorted(zip(read_time, hist), reverse=True, key=lambda x: x[0])
        sorted_read_time, sorted_hist = zip(*sorted_pairs)

        for newsid, rt in zip(sorted_hist, sorted_read_time):
            newsid = str(newsid)
            if newsid not in newsid_vocab.stoi:
                continue
            news_index = newsid_vocab.stoi[newsid]
            if news_index not in news_dict:
                news_dict[news_index] = []
                news_dict_rt[news_index] = []

            user_dict[user_index].append(news_index)
            news_dict[news_index].append(user_index)
            news_dict_rt[news_index].append(rt)
    
    f_user = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"{part}-user_one_hops_rt.txt")
    f_news = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"{part}-news_one_hops_rt.txt")

    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join(map(str, news_list[:cfg.max_user_one_hop]))
            fw.write(f"{user}\t{news_list_str}\n")
    
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            sorted_pairs = sorted(zip(news_dict_rt[news], user_list), reverse=True, key=lambda x: x[0])
            sorted_hist = [user for _, user in sorted_pairs]
            user_list_str = ",".join(map(str, sorted_hist[:cfg.max_news_one_hop]))
            fw.write(f"{news}\t{user_list_str}\n")
    
    return user_dict, news_dict


def main(cfg):
    # f_train_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/behaviors.tsv")
    # f_test_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/behaviors.tsv")
    # f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/history.parquet")
    # f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/history.parquet")
    # f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behavior_histories.parquet")
    # f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behavior_histories.parquet")
    f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/readtime_histories.parquet")
    f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/readtime_histories.parquet")
    # f_test_behaviors = os.path.join(ROOT_PATH, "data/ebnerd_testset/ebnerd_testset", "test/behaviors.parquet")
    train_behavior = pd.read_parquet(f_train_behaviors)
    dev_behavior = pd.read_parquet(f_dev_behaviors)
    # test_behavior = pd.read_parquet(f_test_behaviors)

    f_news_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "newsid_vocab.bin")
    f_user_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "userid_vocab.bin")

    # Load vocab
    user_vocab = WordVocab.load_vocab(f_user_vocab)
    newsid_vocab = WordVocab.load_vocab(f_news_vocab)

    # print(train_behavior['article_id_fixed'])

    # train_behavior = pd.read_csv(f_train_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    # train_behavior = train_behavior.fillna("")
    # train_behavior = train_behavior[train_behavior["hist"]!=""].drop_duplicates("uid")
    # test_behavior = pd.read_csv(f_test_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    # test_behavior = test_behavior.fillna("")
    # test_behavior = test_behavior[test_behavior["hist"]!=""].drop_duplicates("uid")
    def fix_col_names(df):
        df = df.rename(columns={"user_id": "uid", "article_id_fixed": "hist"})
        df['uid'] = df['uid'].astype(str)
        df['hist'] = df['hist'].fillna("")
        df = df.drop_duplicates("uid")
        return df

    train_behavior = fix_col_names(train_behavior)
    dev_behavior = fix_col_names(dev_behavior)
    # test_behavior = fix_col_names(test_behavior)

    if cfg.sort_one_hop_by_read_time:
        train_user_one_hop, train_news_one_hop = build_one_hop_neighbors_rt(cfg, train_behavior, user_vocab, newsid_vocab, "train")
        dev_user_one_hop, dev_news_one_hop = build_one_hop_neighbors_rt(cfg, dev_behavior, user_vocab, newsid_vocab, "validation")
    else:
        train_user_one_hop, train_news_one_hop = build_one_hop_neighbors(cfg, train_behavior, user_vocab, newsid_vocab, "train")
        dev_user_one_hop, dev_news_one_hop = build_one_hop_neighbors(cfg, dev_behavior, user_vocab, newsid_vocab, "validation")
        
    if cfg.rank_two_hop_by_common_clicks:
        build_two_hop_neighbors_ranked(cfg, train_user_one_hop, train_news_one_hop, "train")
        build_two_hop_neighbors_ranked(cfg, dev_user_one_hop, dev_news_one_hop, "validation")
    else:
        build_two_hop_neighbors(cfg, train_user_one_hop, train_news_one_hop, "train")
        build_two_hop_neighbors(cfg, dev_user_one_hop, dev_news_one_hop, "validation")

    # # Test one- and two-hops
    # test_user_one_hop, test_news_one_hop = build_one_hop_neighbors(cfg, test_behavior, user_vocab, newsid_vocab, "test")
    # build_two_hop_neighbors(cfg, test_user_one_hop, test_news_one_hop, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--fvocab", default="vocabs", type=str,
                        help="Path of the training data file.")
    parser.add_argument("--max_user_one_hop", default=50, type=int,
                        help="Maximum number of user one-hop neighbors.")
    parser.add_argument("--max_news_one_hop", default=50, type=int,
                        help="Maximum number of news one-hop neighbors.")
    parser.add_argument("--max_user_two_hop", default=15, type=int,
                        help="Maximum number of user two-hop neighbors.")
    parser.add_argument("--max_news_two_hop", default=15, type=int,
                        help="Maximum number of news two-hop neighbors.")
    parser.add_argument("--sort_one_hop_by_read_time", action='store_true', 
                        help="Whether to sort one-hops by read time.")
    parser.add_argument("--rank_two_hop_by_common_clicks", action='store_true', 
                        help="Whether to use sorted two-hops by common clicks.")

    args = parser.parse_args()

    main(args)

# /home/scur1584/.conda/envs/recsys/bin/python /home/scur1584/GERL/src/scripts/build_neighbors.py --sort_one_hop_by_read_time --rank_two_hop_by_common_clicks