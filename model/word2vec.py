import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from utils import *

def clean(review, remove_freq = True):
    """Given a review, cleans it by removing html tags and punctation.
    Also filters out very common words if the remove_freq flag is given.
    Params:
    review - a single review from the dataset
    remove_freq: True if we remove frequent words (enabled by default)
    """
    # remove html
    text = BeautifulSoup(review).get_text()
    # regexp matching to extract letters only
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] if remove_freq else words
    return (" ".join(meaningful_words))

def create_vocab(cleaned_reviews):
    """Given the cleaned reviews, return a list of all words (without dups)
    that are in the reviews
    """
    li = []
    for review in cleaned_reviews:
        a = review.split()
        for item in a:
            li.append(item)
    return list(set(li))

def get_word_occ_dict(review):
    """Given a review, returns a dictionary of word: number of occurences
    in the review.
    Params: review - a cleaned review from the dataset
    """
    d = defaultdict(int)
    words = review.split()
    for w in words:
        d[w]+=1
    return d

def create_feature_vector(review, vocab):
    """Given a (cleaned) review and a vocabulary, creates a vector of features for that review
    Params:
    review: a cleaned review from the dataset
    vocab: the vocabulary list created with creat_vocab()
    """
    word_dict = get_word_occ_dict(review)
    feature_vector = [word_dict[v] if v in word_dict else 0 for v in vocab]
    return np.array(feature_vector)

def create_feature_vectors(cleaned_reviews, vocab):
    feature_vectors = [create_feature_vector(review, vocab) for review in cleaned_reviews]
    return np.array(feature_vectors)

if __name__ == '__main__':
    train_data = pd.read_csv('../data/labeledTrainData.tsv', header = 0,
                               delimiter = '\t', quoting = 3)
    test_data = pd.read_csv('../data/unlabeledTrainData.tsv', header = 0,
                                delimiter = '\t', quoting = 3)

    train_cleaned_reviews = [clean(train_data["review"][i]) for i in range(len(train_data["review"]))]
    test_cleaned_reviews = [clean(test_data["review"][i]) for i in range(len(test_data["review"]))]

    y = train_data['sentiment']

    vocab = create_vocab(train_cleaned_reviews)
    X = create_feature_vectors(train_cleaned_reviews, vocab)
    print(X.shape)
    print(y.shape)
