# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/2

Some help functions for building dataset
"""
import os
import re
import json
from typing import List, Dict, Set

import pandas as pd


PADDING_NEWS = "<pad>"
ROOT_PATH = os.environ["GERL"]


def word_tokenize_old(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

import spacy

# Load the Danish language model
nlp = spacy.load("da_core_news_sm")

def word_tokenize(sent):
    """ Tokenize sentence using spacy.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    doc = nlp(sent)
    return [token.text.lower() for token in doc]

