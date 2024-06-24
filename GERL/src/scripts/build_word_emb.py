import os
import json
import pickle
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import fasttext
import fasttext.util

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.vocab import WordVocab
from utils.build_util import word_tokenize

ROOT_PATH = os.environ["GERL"]

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
            token = row[0]
            # token = ' '.join(word_tokenize(row[0]))  # Apply the same tokenizer to the embedding keys
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
    # np.save(weights_output_file, weights_matrix)

def build_word_embeddings_bin(vocab, pretrained_embedding, weights_output_file):
    # Load embeddings from FastText binary file
    ft_model = fasttext.load_model(pretrained_embedding)
    embed_size = ft_model.get_dimension()
    vocab_size = len(vocab)

    # Initialize the weight matrix
    weights_matrix = np.zeros((vocab_size, embed_size))
    words_found = 0

    # Fill the weights matrix
    for i, word in enumerate(vocab.itos):
        weights_matrix[i] = ft_model.get_word_vector(word)
        words_found += 1
        # Note: FastText model returns vectors for out-of-vocabulary words as well

    print(f"Totally found {words_found} words in pre-trained embeddings.")

    # Save the weights matrix
    np.save(weights_output_file, weights_matrix)

def main(cfg):
    f_word_embeddings = os.path.join(ROOT_PATH, "data", cfg.word_embeddings)
    f_word_vocab = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.fvocab, "word_vocab.bin")

    # Load vocab
    word_vocab = WordVocab.load_vocab(f_word_vocab)

    # Build word embeddings
    print("Building word embedding matrix")
    weight_path = os.path.join(ROOT_PATH, "data", f"ebnerd_{cfg.fsize}", cfg.output, "word_embeddings.npy")

    # Check the file extension and call the appropriate function
    if f_word_embeddings.endswith(".bin"):
        build_word_embeddings_bin(word_vocab, f_word_embeddings, weight_path)
    elif f_word_embeddings.endswith(".vec"):
        build_word_embeddings(word_vocab, f_word_embeddings, weight_path)
    else:
        raise ValueError("Unsupported file extension. Only .vec and .bin are supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="demo", type=str, help="Corpus size")
    parser.add_argument("--fvocab", default="vocabs", type=str, help="Path of the vocabs dir.")
    parser.add_argument("--word_embeddings", default="cc.da.300.bin", type=str, help="Path of the embeddings.")
    parser.add_argument("--output", default="vocabs", type=str, help="Path of the output.")

    args = parser.parse_args()

    main(args)
