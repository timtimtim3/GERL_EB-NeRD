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


def build_img_embeddings(news_vocab, img_embeddings, weights_output_file, emb_col):
    # Create a dictionary for quick access to embeddings using article_id
    emb_dict = dict(zip(img_embeddings['article_id'].astype(str), img_embeddings[emb_col]))

    embed_size = len(next(iter(emb_dict.values())))
    vocab_size = len(news_vocab)

    # Initialize the weight matrix with zeros
    weights_matrix = np.zeros((vocab_size, embed_size))
    docs_found = 0
    docs_not_found = set()

    for article_id, index in news_vocab.stoi.items():
        try:
            weights_matrix[index] = emb_dict[article_id]
            docs_found += 1
        except KeyError:
            docs_not_found.add(article_id)
            weights_matrix[index] = np.random.normal(size=(embed_size,))
    
    print(f"Totally found {docs_found} documents in pre-trained embeddings.")
    print(f"Totally did not find {len(docs_not_found)} documents in pre-trained embeddings.")

    # Save the weight matrix
    np.save(weights_output_file, weights_matrix)


def main(cfg):         
    f_img_embeddings = os.path.join(ROOT_PATH, "data", "Ekstra_Bladet_image_embeddings", "Ekstra_Bladet_image_embeddings", "image_embeddings.parquet")
    f_news_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "newsid_vocab.bin")

    # Load vocab
    img_embeddings = pd.read_parquet(f_img_embeddings)
    news_vocab = WordVocab.load_vocab(f_news_vocab)

    print(img_embeddings.head())

    # Build word embeddings
    print("Building img embedding matrix")
    weight_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, f"img_embeddings.bin")
    emb_col = "image_embedding"

    build_img_embeddings(news_vocab, img_embeddings, weight_path, emb_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options. 
    parser.add_argument("--fsize", default="small", type=str,
                        help="Corpus size")
    parser.add_argument("--fvocab", default="vocabs", type=str,
                    help="Path of the vocabs dir.")
    parser.add_argument("--output", default="vocabs", type=str,
                        help="Path of the output.")

    args = parser.parse_args()

    main(args)
