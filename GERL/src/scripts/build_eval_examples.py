# -*- coding: utf-8 -*-
"""Script for building the eval examples.
{
    imp_id: 000,
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: 7,
    y: 1
    neighbor_news: [27, 28, 29]
}
"""
import os
import json
import random
import argparse
import sys
from typing import List, Dict

import tqdm
import pandas as pd

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.vocab import WordVocab

random.seed(7)
ROOT_PATH = os.environ["GERL"]


def build_examples(cfg, df: pd.DataFrame, 
                   user_vocab: WordVocab, newsid_vocab: WordVocab, 
                   user_one_hop: Dict, news_one_hop: Dict, 
                   user_two_hop: Dict, news_two_hop: Dict, subsample=False, subsample_size=20000):
    
    def _get_neighbors(neighbor_dict, key, max_neighbor_num, pad=False):
        neighbors = neighbor_dict.get(key, [])
        if pad:
            # Pad neighbors to the required length
            if len(neighbors) < max_neighbor_num:
                neighbors.extend([0] * (max_neighbor_num - len(neighbors)))  # Assuming 0 is the pad_index
            else:
                neighbors = neighbors[:max_neighbor_num]
        return neighbors


    # Adjust output directory based on the flags
    output_dir_suffix = "examples"
    if cfg.sort_one_hop_by_read_time and cfg.rank_two_hop_by_common_clicks:
        output_dir_suffix = "examples_rt_ranked"
    elif cfg.rank_two_hop_by_common_clicks:
        output_dir_suffix = "examples_ranked"
    elif cfg.sort_one_hop_by_read_time:
        output_dir_suffix = "examples_rt"

    f_out_dir = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", output_dir_suffix)
    os.makedirs(f_out_dir, exist_ok=True)

    if subsample:
        f_out = os.path.join(f_out_dir, "eval_examples_subsample.tsv")
    else:
        f_out = os.path.join(f_out_dir, "eval_examples.tsv")
    fw = open(f_out, "w", encoding="utf-8")

    df['uid'] = df['uid'].astype(str)
    df['article_ids_clicked'] = df['article_ids_clicked'].astype(str)
    df['impression_id'] = df['impression_id'].astype(str)

    df = df.groupby('uid')

    user_count = 0
    for uid, group in df:

        if subsample and user_count >= subsample_size:
            break

        user_index = user_vocab.stoi.get(uid, 0)
        hist_news = _get_neighbors(user_one_hop, user_index, cfg.max_user_one_hop)
        neighbor_users = _get_neighbors(user_two_hop, user_index, cfg.max_user_two_hop)
        
        for index, row in group.iterrows():
            # news_id, label = sample.split("-")[:2]
            neighbor_news = []
            target_news = [newsid_vocab.stoi.get(row['article_ids_clicked'], 0)] + [newsid_vocab.stoi.get(str(x), 0) for x in row['negative_pool']]
            for news_index in target_news:
                # hist_users.append(_get_neighors(news_one_hop, news_index, cfg.max_news_one_hop))
                neighbor_news.append(_get_neighbors(news_two_hop, news_index, cfg.max_news_two_hop))

            for i, (target, n_news) in enumerate(zip(target_news, neighbor_news)):
                j = {
                    "user": user_index,
                    "hist_news": hist_news,
                    "neighbor_users": neighbor_users,
                    "target_news": target,
                    # "hist_users": hist_users,
                    "neighbor_news": n_news,
                    "y": 1 if i == 0 else 0,
                    "imp_id": int(row['impression_id'])
                }
                fw.write(json.dumps(j) + "\n")

        user_count += 1


def load_hop_dict(fpath: str) -> Dict:
    lines = open(fpath, "r", encoding="utf-8").readlines()
    d = dict()
    error_line_count = 0
    for line in lines:
        row = line.strip().split("\t")
        if len(row) != 2:
            error_line_count += 1
            continue
        key, vals = row[:2]
        vals = [int(x) for x in vals.split(",")]
        d[int(key)] = vals
    print("{} error lines: {}".format(fpath, error_line_count))
    return d

def main(cfg):
    f_dev_samples = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", "validation/samples.parquet")
    f_news_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "newsid_vocab.bin")
    f_user_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "userid_vocab.bin")
    
    # Adjust paths based on the flags
    one_hop_suffix = "_rt" if cfg.sort_one_hop_by_read_time else ""
    two_hop_suffix = "_ranked" if cfg.rank_two_hop_by_common_clicks else ""
    if cfg.sort_one_hop_by_read_time and cfg.rank_two_hop_by_common_clicks:
        two_hop_suffix = "_rt_ranked"

    f_user_one_hop = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"validation-user_one_hops{one_hop_suffix}.txt")
    f_news_one_hop = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"validation-news_one_hops{one_hop_suffix}.txt")
    f_user_two_hop = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"validation-user_two_hops{two_hop_suffix}.txt")
    f_news_two_hop = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, f"validation-news_two_hops{two_hop_suffix}.txt")

    # Load vocab
    user_vocab = WordVocab.load_vocab(f_user_vocab)
    newsid_vocab = WordVocab.load_vocab(f_news_vocab)
    user_one_hop = load_hop_dict(f_user_one_hop)
    news_one_hop = load_hop_dict(f_news_one_hop)
    user_two_hop = load_hop_dict(f_user_two_hop)
    news_two_hop = load_hop_dict(f_news_two_hop)

    # 预处理好的训练样本
    # samples = pd.read_csv(f_dev_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    samples = pd.read_parquet(f_dev_samples)

    build_examples(cfg, samples, user_vocab, newsid_vocab, user_one_hop, news_one_hop, user_two_hop,
                   news_two_hop, subsample=cfg.subsample, subsample_size=cfg.subsample_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--fout", default="hop1_cocur_bip_hist50", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--fvocab", default="vocabs", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--max_user_one_hop", default=50, type=int,
                        help="Maximum number of user one-hop neighbors.")
    parser.add_argument("--max_news_one_hop", default=50, type=int,
                        help="Maximum number of news one-hop neighbors.")
    parser.add_argument("--max_user_two_hop", default=15, type=int,
                        help="Maximum number of user two-hop neighbors.")
    parser.add_argument("--max_news_two_hop", default=15, type=int,
                        help="Maximum number of news two-hop neighbors.")
    parser.add_argument("--subsample", action='store_true', help="Flag to subsample users.")
    parser.add_argument("--subsample_size", default=5000, type=int,
                        help="Maximum number of news two-hop neighbors.")
    parser.add_argument("--sort_one_hop_by_read_time", action='store_true', 
                        help="Whether to use sorted one-hops by read time.")
    parser.add_argument("--rank_two_hop_by_common_clicks", action='store_true', 
                        help="Whether to use sorted two-hops by common clicks.")

    args = parser.parse_args()

    main(args)

# /home/scur1584/.conda/envs/recsys/bin/python /home/scur1584/GERL/src/scripts/build_eval_examples.py --sort_one_hop_by_read_time --rank_two_hop_by_common_clicks