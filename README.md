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

You can call `python GERL/src/scripts/build.py` to pre-process the data, build the vocabs, indices, embedding matrices, one-hop and two-hop neighbors, and the train and validation examples all at once. This script calls all the necessary `scripts` in the scripts folder.

You can also run the scripts individually with additional arguments. For example, you might want to build the neighbors (`build_neighbors.py`) with user and news one-hops sorted by read-time and two-hops ranked by commonly clicked articles (as done in the original paper, but not in the un-official codebase) by setting the `sort_one_hop_by_read_time` and `rank_two_hop_by_common_clicks` to True, respectively.

After this, you can call `train.py` to train a model.

## Hydra

In `GERL/src/conf` there are folders and yaml files for the configuration of the train and test runs. Here is an overview of the arguments we added or regularly used:

training.use_doc_embeddings: Set to True if you want to use EB-NeRD document embeddings instead of word embeddings.
training.use_img_embeddings: Set to True if you want to use EB-NeRD image embeddings (keep training.use_doc_embeddings to False, this is still a work-in-progress).

model.name: Set to "word_emb", "word2vec", "facebook_roberta", or "google_bert_multilingual".

dataset.sort_one_hop_by_read_time: Set to True if you sorted one-hops by read time and want to use those neighbor files.
dataset.rank_two_hop_by_common_clicks: Set to True if you ranked user two-hops using common clicks and want to use those neighbor files.
dataset.size: Pick "ebnerd_demo", "ebnerd_small", or "ebnerd_large".
dataset.doc_emb_kind: Set to "word_emb", "word2vec", "facebook_roberta", or "google_bert_multilingual".
dataset.name: Set to "examples", "examples_rt", "examples_ranked", or "examples_rt_ranked". We use "examples" (regular neighbors) and "examples_rt_ranked" (sorted by readtime and ranked by common clicks).
dataset.valid_name: Pick "eval_examples_subsample.tsv" if you created a validation subsample with `build_eval_examples.py` (see section "What we added") otherwise use default ("eval_examples.tsv").

### Example train runs:

*Regular neighbors with word embeddings:*
python -u GERL/src/train.py model.name="word_emb" training.epochs=10 dataset.size="ebnerd_small" training.use_doc_embeddings=False

*Regular neighbors with document embeddings*:
python -u GERL/src/train.py model.name="facebook_roberta" training.epochs=10 dataset.size="ebnerd_small" training.use_doc_embeddings=True dataset.valid_name="eval_examples_subsample.tsv" doc_emb_kind="facebook_roberta"

*Sorted and ranked neighbors with word embeddings:*
python -u GERL/src/train.py dataset.name="examples_rt_ranked" model.name="word_emb" training.epochs=10 dataset.size="ebnerd_small" training.use_doc_embeddings=False dataset.valid_name="eval_examples_subsample.tsv" dataset.sort_one_hop_by_read_time=True dataset.rank_two_hop_by_common_clicks=True

*Image embeddings (work-in-progress):*
python -u GERL/src/train.py dataset.name="examples" model.name="word_emb" training.epochs=10 dataset.size="ebnerd_small" training.use_doc_embeddings=False dataset.valid_name="eval_examples_subsample.tsv" training.use_img_embeddings=True
python -u GERL/src/train.py dataset.name="examples_rt_ranked" model.name="word_emb" training.epochs=10 dataset.size="ebnerd_small" training.use_doc_embeddings=False dataset.valid_name="eval_examples_subsample.tsv" dataset.sort_one_hop_by_read_time=True dataset.rank_two_hop_by_common_clicks=True training.use_img_embeddings=True

### Example test runs:

*Regular neighbors with word embeddings:*
python -u GERL/src/test.py model.name="word_emb" training.validate_epoch=10 dataset.size="ebnerd_small"

*Regular neighbors with document embeddings*:
srun python -u GERL/src/test.py model.name="facebook_roberta" training.validate_epoch=10 dataset.size="ebnerd_small" training.use_doc_embeddings=True doc_emb_kind="facebook_roberta"

*Sorted and ranked neighbors with word embeddings:*
python -u GERL/src/test.py dataset.name="examples_rt_ranked" model.name="word_emb" training.validate_epoch=10 dataset.size="ebnerd_small" dataset.sort_one_hop_by_read_time=True dataset.rank_two_hop_by_common_clicks=True

## What we added

In `GERL/src/scripts`:

Added: `extract_behavior_histories.py` which extracts the clicked article histories of users.

Added: `extract_read_times.py` which extracts the clicked article histories of users as well as the corresponding read times. Use this instead of `extract_behavior_histories.py` if you want to order one-hops by read time. 

Added: `extract_samples.py` which samples four negative news articles for each impression in the dataset from the set of inview articles in that impression minus all clicked articles for that user. So each negative sample was seen by the user in that impression, but was never clicked by the user.

Changed: `build_vocabs.py`, we altered this script to work with the EB-NeRD dataset and to build the word and user vocabs and the news-id to title index. We use a Danish word tokenizer to tokenize the news titles. This script no longer builds the word embedding matrix because we added a seperate script for that.

Added: `build_word_emb.py` which loads a [Danish FastText](https://fasttext.cc/docs/en/crawl-vectors.html) word embedding binary and saves the word embeddings an an embedding matrix. Make sure to download this binary if you want to use word embeddings instead of the document embeddings available in EB-NeRD.

Added: `build_doc_emb.py` loads the specified document embeddings (specify word2vec, facebook_roberta, or google_bert_multilingual for the `doc_embeddings` argument) and builds the document embedding matrix where each line/index corresponds to a news article with that index. Make sure the vocabs are built first with `build_vocabs.py`.

Changed: `build_neighbors.py` to work with EB-NeRD data and added two boolean arguments `sort_one_hop_by_read_time` and `rank_two_hop_by_common_clicks` which can set to True to sort user and news one-hops by read-time and rank two-hops by commonly clicked articles between users, respectively.

Changed: `build_training_examples.py` and `build_eval_examples.py` to work with the EB-NeRD dataset and to save the .tsv training/validation files in different folders depending on whether you're using `sort_one_hop_by_read_time` or `rank_two_hop_by_common_clicks` or a combination of the two. We also added a `subsample` and `subsample_size` flag to allow subsampling the validation set since the EB-NeRD dataset has a validation set with similar size to the training set which can take an excessive amount of time to validate on. We suggest you subsample for training, and then validate on either the full set afterwards or use a subset as well in `test.py`.

Added: `build_image_emb.py` which extracts the EB-NeRD image embeddings and creates an image embedding matrix using the news vocab which can be used during training.

Changed: In `GERL/src/models/gerl.py`, we added a `ModelDocEmb` class which functions similarly to the original `Model` class but loads pretrained document embeddings as trainable parameters and uses these in place of the `Transformer` models that encode the news titles in the original paper. These document embeddings are projected down to match the size of the other representations (128) with a linear layer.

Changed: We adapted `train.py` and `test.py` so they create a normal model or a document embedding model based on the config. We adusted them to be able to differentiate between different graphs "_rt" (one-hops sorted by read time), "_ranked" (two-hops ranked by common clicks), "_rt_ranked" (both), and the regular graph. All this does is loading the correct neighbor files and saving the models and metrics in different folders. 

Work-in-progress: In `GERL/src/models/gerl.py`, we adapted the `Model` class to allow for including the EB-NeRD pre-trainedd image embeddings for each news article. The idea is to index into the learnable image embedding matrix to get the image embeddings for the Neighbor News (news two-hops), Clicked News (user one-hops), and Candidate News and project the embeddings down to 128 using a learnt linear projection. We then add a `SelfAttendLayer`, one for the news image two-hops and one for the user image one-hops, to attend over the different image ebeddings to get a 128-sized news two-hop image representation and a 128-sized user one-hop image representation. We then append the news two-hop image representation and the candidate news image embedding to a list of existing news representations (which contains three elements in the paper) and aggregate over those five 128-sized vectors e.g. by summing them. We add the user one-hop image representation to the list of user representations and aggregate those four 128-sized vectors using the same method e.g. by summing.

Work-in-progress: In `GERL/src/models/gerl.py`, we adapted the `Model` class to allow for additional aggregation methods for the different user representions and news representations. This includes summing the different news or user representations (as in the original paper), but we also allow for passing the representations through an MLP instead so that the final aggregation may be learned, and allow as another option a Multi-Head attention network coupled with a projection layer that projects the user or news representation to the right size (128). Finally, we take the dot product between the news and user representations. The MLP and Multi-Head attention aggregation methods are a work in progress so you should use adding for now (default).
