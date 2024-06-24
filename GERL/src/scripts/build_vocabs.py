# -*- coding: utf-8 -*-
"""
构建dict：
- news_id => news index
- user_id => user index
- word => word index
- news_index => title seq

news_id 和 user_id ，word 只用出现在train中的
"""
import os
import json
import pickle
import argparse
import sys
import polars as pl
import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from datasets.vocab import WordVocab
from utils.build_util import word_tokenize

ROOT_PATH = os.environ["GERL"]

def build_word_embeddings_old(vocab, pretrained_embedding, weights_output_file):
    # Load embeddings
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()

    emb_dict = dict()
    error_line = 0
    embed_size = int(lines[0].strip().split()[1])
    print(lines[0])
    print()
    print(lines[1])
    print()

    for i, line in enumerate(lines[1:]):
        row = line.strip().split()
        token = ' '.join(word_tokenize(row[0]))  # Apply the same tokenizer to the embedding keys
        try:
            embedding = np.array(row[1:]).astype(float)
            emb_dict[token] = embedding
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    # embed_size = len(emb_dict.values()[0])
    # build embedding weights for model
    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0
    words_not_found = set()

    for i, word in enumerate(vocab.itos):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except KeyError:
            words_not_found.add(word)
            weights_matrix[i] = np.random.normal(size=(embed_size,))
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    print(words_not_found)
    print("Totally did not find {} words in pre-trained embeddings.".format(len(words_not_found)))
    np.save(weights_output_file, weights_matrix)


def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Initialize variables
    emb_dict = {}
    error_line = 0

    # Process the lines into embeddings dictionary
    with open(pretrained_embedding, "r", encoding="utf8") as f:
        # Read the first line to get the embedding size
        first_line = f.readline().strip().split()
        embed_size = int(first_line[1])
        
        for line in f:
            row = line.strip().split()
            token = ' '.join(word_tokenize(row[0]))  # Apply the same tokenizer to the embedding keys
            try:
                embedding = np.array(row[1:], dtype=float)
                emb_dict[token] = embedding
            except ValueError:
                error_line += 1

    print("Error lines: {}".format(error_line))

    # Initialize the weight matrix
    vocab_size = len(vocab)
    weights_matrix = np.zeros((vocab_size, embed_size))
    words_found = 0
    words_not_found = set()

    # Fill the weights matrix
    for i, word in enumerate(vocab.itos):
        if word in emb_dict:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        else:
            words_not_found.add(word)
            weights_matrix[i] = np.random.normal(size=(embed_size,))

    print("Totally found {} words in pre-trained embeddings.".format(words_found))
    print("Totally did not find {} words in pre-trained embeddings.".format(len(words_not_found)))
    print(words_not_found)

    # Save the weights matrix
    np.save(weights_output_file, weights_matrix)


def build_user_id_vocab(cfg, behavior_df):
    user_vocab = WordVocab(behavior_df.uid.values, max_size=1000000, min_freq=1, lower=cfg.lower)
    print("USER ID VOCAB SIZE: {}".format(len(user_vocab)))
    f_user_vocab_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, "userid_vocab.bin")
    user_vocab.save_vocab(f_user_vocab_path)
    return user_vocab

def build_news_id_vocab(cfg, news_df):
    news_vocab = WordVocab(news_df.newsid.values, max_size=140000, min_freq=1, lower=cfg.lower)
    print("NEWS ID VOCAB SIZE: {}".format(len(news_vocab)))
    f_news_vocab_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, "newsid_vocab.bin")
    news_vocab.save_vocab(f_news_vocab_path)
    return news_vocab

def build_word_vocab(cfg, news_df):
    word_vocab = WordVocab(news_df.title_token.values, max_size=cfg.size, min_freq=1, lower=cfg.lower)
    print("TEXT VOCAB SIZE: {}".format(len(word_vocab)))
    f_text_vocab_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, "word_vocab.bin")
    word_vocab.save_vocab(f_text_vocab_path)
    return word_vocab

def build_newsid_to_title(cfg, news_df: pd.DataFrame, newsid_vocab: WordVocab, word_vocab: WordVocab):
    news2title = np.zeros((len(newsid_vocab) + 1, cfg.max_title_len), dtype=int)
    news2title[0] = word_vocab.to_seq('<pad>', seq_len=cfg.max_title_len)
    for row in news_df[["newsid", "title_token"]].values:
        news_id, title = row[:2]
        news_index = newsid_vocab.stoi[news_id]
        news2title[news_index], cur_len = word_vocab.to_seq(title, seq_len=cfg.max_title_len, with_len=True)
    
    f_title_matrix = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", "news_title.npy")
    np.save(f_title_matrix, news2title)
    print("title embedding: ", news2title.shape)

def main(cfg):         
    os.makedirs(os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output), exist_ok=True)

    # Build vocab
    print("Loading news info")
    # f_train_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/news.tsv")
    # f_dev_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "dev/news.tsv")
    # f_test_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/news.tsv")

    f_train_dev_news = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "articles.parquet")
    f_test_news = os.path.join(ROOT_PATH, "data/ebnerd_testset/ebnerd_testset", "articles.parquet")

    print("Loading training news")
    # train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
    #                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
    #                        quoting=3)
    # if os.path.exists(f_dev_news):
    #     print("Loading dev news")
    #     dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
    #                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
    #                            quoting=3)
    # if os.path.exists(f_test_news):
    #     print("Loading testing news")
    #     test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
    #                             names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
    #                             quoting=3)
        
    train_dev_news = pd.read_parquet(f_train_dev_news)
    test_news = pd.read_parquet(f_test_news)

    print(train_dev_news.head())
    print(train_dev_news.columns)

    train_dev_news = train_dev_news.rename(columns={"article_id": "newsid"})
    test_news = test_news.rename(columns={"article_id": "newsid"})

    # all_news = pd.concat([train_news, dev_news, test_news], ignore_index=True)
    all_news = pd.concat([train_dev_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates("newsid")
    print("All news: {}".format(len(all_news)))

    # 单独处理train news
    # train_news['title_token'] = train_news['title'].apply(lambda x: ' '.join(word_tokenize(x)))
    all_news['title_token'] = all_news['title'].apply(lambda x: ' '.join(word_tokenize(x)))

    # Build user id vocab
    # f_train_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/behaviors.tsv")
    # f_dev_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "dev/behaviors.tsv")
    # f_test_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/behaviors.tsv")
    # train_behavior = pd.read_csv(f_train_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    # dev_behavior = pd.read_csv(f_dev_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    # test_behavior = pd.read_csv(f_test_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    
    f_train_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "train/behaviors.parquet")
    f_dev_behaviors = os.path.join(ROOT_PATH, f"data/ebnerd_{cfg.fsize}", "validation/behaviors.parquet")
    f_test_behaviors = os.path.join(ROOT_PATH, "data/ebnerd_testset/ebnerd_testset", "test/behaviors.parquet")
    train_behavior = pd.read_parquet(f_train_behaviors)
    dev_behavior = pd.read_parquet(f_dev_behaviors)
    test_behavior = pd.read_parquet(f_test_behaviors)

    behaviors_df = pd.concat([train_behavior, dev_behavior, test_behavior], ignore_index=True)
    behaviors_df = behaviors_df.rename(columns={"user_id": "uid"})
    behaviors_df = behaviors_df.drop_duplicates("uid")

    behaviors_df['uid'] = behaviors_df['uid'].astype(str)
    all_news['newsid'] = all_news['newsid'].astype(str)

    _ = build_user_id_vocab(cfg, behaviors_df) # uses uid only
    
    # Build news id vocab, uses newsid only
    newsid_vocab = build_news_id_vocab(cfg, all_news)

    # Build word vocab, uses title_token only which is titles tokenized
    # word_vocab = build_word_vocab(cfg, train_news)
    word_vocab = build_word_vocab(cfg, all_news)

    # Build word embeddings
    print("Building word embedding matrix")
    pretrain_path = os.path.join(ROOT_PATH, cfg.pretrain)
    weight_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, "word_embeddings.bin")
    # build_word_embeddings(word_vocab, pretrain_path, weight_path)

    # Build news_index => title word seq
    build_newsid_to_title(cfg, all_news, newsid_vocab, word_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options. 
    parser.add_argument("--fsize", default="large", type=str,
                        help="Corpus size")
    # parser.add_argument("--pretrain", default="data/glove.840B.300d.txt", type=str,
    #                     help="Path of the raw review data file.")
    parser.add_argument("--pretrain", default="data/cc.da.300.vec", type=str,
                    help="Path of the raw review data file.")
    parser.add_argument("--output", default="vocabs", type=str,
                        help="Path of the training data file.")
    parser.add_argument("--size", default=80000, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--max_title_len", default=30, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--lower", action='store_true')

    args = parser.parse_args()

    main(args)
