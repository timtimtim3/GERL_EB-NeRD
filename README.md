# GERL_EB-NeRD

## Introduction
This project does adapted the un-official [GERL](https://github.com/zpqiu/GERL/tree/main) implementation by zpqui of the paper "Graph Enhanced Representation Learning for News Recommendation" to work wit the Ekstra Bladet News Recommendation Dataset (EB-NeRD). 
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

You can call `python GERL/src/scripts/build.py` to pre-process the data, build the vocabs, indices, and embedding matrices, the neighbors, and the train and validation examples all at once. This script calls all the necessary `scripts` in the scripts folder.

You can also run the scripts individually with additional arguments. For example, you might want to build the neighbors (`build_neighbors.py`) with user and news one-hops sorted by read-time and two-hops ranked by commonly clicked articles (as done in the original paper, but not in the un-official codebase) by setting the `sort_one_hop_by_read_time` and `rank_two_hop_by_common_clicks` to True, respectively.

 After this, you can call `train.py` to train a model.


## What was added
