import numpy as np
import tensorflow as tf
import pickle

def generate_review_features(review, word_to_idx_dict, embeddings):
    x = []
    for word in review.split():
        embedding = embeddings[word_to_idx_dict[word]]
if __name__ == '__main__':
    # the following code just reads in the word embeddings and a mapping from word to vector.
    try:
        print("loading embeddings")
        with open('../tf-implementation/data/embeddings.txt', 'rb') as f:
            final_embeddings, reverse_dictionary = pickle.load(f)
    except IOError:
        print("please run word2vec.py which will learn and save the embeddings.")
    try:
        print("loading data")
        with open('../tf-implementation/data/data.txt', 'rb') as f:
            train_cleaned_reviews, test_cleaned_reviews, vocab, y = pickle.load(f)
            print(type(y))
            print("done loading")
    except IOError:
        print("please run word2vec.py which will load and save the data")

    # variables needed later
    num_clusts = 500
    word_to_idx_dict = {v: k for k, v in reverse_dictionary.items()}
    word = list(word_to_idx_dict.keys())[15]
    print("Word {} has index {} which coressponds to embedding: {}".format(word, word_to_idx_dict[word], final_embeddings[word_to_idx_dict[word]]))
