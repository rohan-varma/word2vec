import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import accuracy_score
sys.path.append('../')
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier

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
    Takes the review and generates a dictionary of word occurences. Then the vector returned has
    sizeof(vocab) dimensions with the ith feature indicating the amount of times word i in the vocab
    occured in the feature.
    Therefore, this feature vector will be a sparse representation
    Params:
    review: a cleaned review from the dataset
    vocab: the vocabulary list created with creat_vocab()
    """
    word_dict = get_word_occ_dict(review)
    feature_vector = [word_dict[v] if v in word_dict else 0 for v in vocab]
    return np.array(feature_vector)

def create_feature_vectors(cleaned_reviews, vocab):
    feature_vectors = [create_feature_vector(review, vocab)
                       for review in cleaned_reviews]
    return np.array(feature_vectors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="load data from data.txt",
                        action = "store_true")
    args = parser.parse_args()

    if args.load:
        print("loading data")
        with open('data.txt', 'rb') as f:
            train_cleaned_reviews, test_cleaned_reviews, vocab, y = pickle.load(f)
            print("done loading")
    else:
        train_data = pd.read_csv('../data/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
        test_data = pd.read_csv('../data/unlabeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
        print("done reading data")
        train_cleaned_reviews = [clean(train_data["review"][i]) for i in range(len(train_data["review"]))]
        test_cleaned_reviews = [clean(test_data["review"][i]) for i in range(len(test_data["review"]))]
        print("done cleaning data")
        y = train_data['sentiment']
        vocab = create_vocab(train_cleaned_reviews)
        with open('data.txt', 'wb') as f:
            print("dumping data")
            pickle.dump([train_cleaned_reviews, test_cleaned_reviews, vocab, y], f, -1)

    print("creating feature vectors")
    X = create_feature_vectors(train_cleaned_reviews, vocab) #EXPENSIVE, remember to pickle this as well.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    clf = LinearSVC(C = 1.0, loss = 'l2')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    test_acc = accuracy_score(y_true=y_test, y_pred=preds)
    acc_file = open('bag-of-words-acc.txt', 'w')
    print("test accuracy for Linear SVM: {}".format(test_acc))
    acc_file.write("test accuracy for Linear SVM: {}".format(test_acc))
    print("running random forest clf")
    rand_forest_clf = RandomForestClassifier()
    rand_forest_clf.fit(X_train, y_train)
    preds = rand_forest_clf.predict(X_test)
    forest_test_acc = accuracy_score(y_true = y_test, y_pred = preds)
    print("test accuracy (using random forest): {}".format(forest_test_acc))
    acc_file.write("test accuracy, using random forest: {}".format(forest_test_acc))
    acc_file.close()
