# GERL_EB-NeRD

## Introduction
This project adapted the un-official [GERL](https://github.com/zpqiu/GERL/tree/main) implementation by zpqui of the paper "Graph Enhanced Representation Learning for News Recommendation" to work wit the Ekstra Bladet News Recommendation Dataset (EB-NeRD). 
This dataset was provided by Ekstra Bladet for the 2024 [Recommender System Challenge](https://recsys.eb.dk/#about). We also extended the GERL code to ...

## Directory Structure
Ensure that you have the following directory structure in your home directory:

```
├── GERL
│   └── (GERL_EB-NeRD files and directories)
└── data
    ├── ebnerd_demo
    │   └── (EB-NeRD demo dataset files)
    ├── ebnerd_small
    │   └── (EB-NeRD small dataset files)
    └── ebnerd_large
        └── (EB-NeRD large dataset files)
```

To run you need a folder "data" in which you should put the "ebnerd_demo", "ebnerd_small", or "ebnerd_large" folders which should contain the EB-NeRD dataset. You can use any size of the dataset. 

## Running

You can call `python GERL/src/scripts/build.py` to pre-process the data, build the vocabs, indices, embedding matrices, one-hop and two-fop neighbors, and the train and validation examples all at once. This script calls all the necessary `scripts` in the scripts folder.

You can also run the scripts individually with additional arguments. For example, you might want to build the neighbors (`build_neighbors.py`) with user and news one-hops sorted by read-time and two-hops ranked by commonly clicked articles (as done in the original paper, but not in the un-official codebase) by setting the `sort_one_hop_by_read_time` and `rank_two_hop_by_common_clicks` to True, respectively.

After this, you can call `train.py` to train a model.

## What we added

Added: `extract_behavior_histories.py` which extracts the clicked article histories of users.

Added: `extract_read_times.py` which extracts the clicked article histories of users as well as the corresponding read times. Use this instead of `extract_behavior_histories.py` if you want to order one-hops by read time. 

Added: `extract_samples.py` which samples four negative news articles for each impression in the dataset from the set of inview articles in that impression minus all clicked articles for that user. So each negative sample was seen by the user in that impression, but was never clicked by the user.

Changed: `build_vocabs.py`, we altered this script to work with the EB-NeRD dataset and to build the word and user vocabs and the news-id to title index. We use a Danish word tokenizer to tokenize the news titles. This script no longer builds the word embedding matrix because we added a seperate script for that.

Added: `build_word_emb.py` which loads a [Danish FastText](https://fasttext.cc/docs/en/crawl-vectors.html) word embedding binary and saves the word embeddings an an embedding matrix. Make sure to download this binary if you want to use word embeddings instead of the document embeddings available in EB-NeRD.

Added: `build_doc_emb.py` loads the specified document embeddings (specify word2vec, facebook_roberta, or google_bert_multilingual for the `doc_embeddings` argument) and builds the document embedding matrix where each line/index corresponds to a news article with that index. Make sure the vocabs are built first with `build_vocabs.py`.




