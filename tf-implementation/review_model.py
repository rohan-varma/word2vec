import numpy as np
import tensorflow as tf
import pickle
import sklearn
from sklearn.cluster import KMeans
from collections import Counter

def embeddings_lookup(embeddings, word_to_idx_dict, word):
    try:
        idx = word_to_idx_dict[word]
    except KeyError:
        return -1
    ret = embeddings[idx]
    assert(ret.shape[0] == 128)
    return ret

def review_to_embedding(embeddings, word_to_idx_dict, review):
    words = review.split()
    embedding_matrix = [embeddings_lookup(embeddings, word_to_idx_dict, word)
                        for word in words]
    embedding_matrix = np.array(embedding_matrix)
    assert(embedding_matrix.shape[1] == 128)
    return embedding_matrix


if __name__ == '__main__':
    try:
        print("loading embeddings")
        with open('embeddings.txt', 'rb') as f:
            final_embeddings, reverse_dictionary = pickle.load(f)
    except IOError:
        print("please run word2vec.py which will learn and save the embeddings.")
    try:
        print("loading data")
        with open('../bag-of-words/data.txt', 'rb') as f:
            train_cleaned_reviews, test_cleaned_reviews, vocab, y = pickle.load(f)
            print(type(y))
            print("done loading")
    except IOError:
        print("please run bwmodel.py which will load and save the data")
    num_clusts = 500
    #print("word 1: {}".format(reverse_dictionary[1]))
    word_to_idx_dict = {v: k for k, v in reverse_dictionary.items()}
    # checking word embeddings on a sample review (expensive)
    # sample_review = train_cleaned_reviews[0]
    # embeds = []
    # for word in sample_review.split():
    #     print("word in use: {}".format(word))
    #     embedding = embeddings_lookup(final_embeddings, word_to_idx_dict, word)
    #     embeds.append(embedding)
    # embeds = np.array(embeds)
    # print(embeds.shape)
    # clustering embeddings
    print("clustering")
    kmeans = KMeans(n_clusters = 500, max_iter = 10000).fit(final_embeddings)
    print("done clustering")
    kmeans.predict(final_embeddings[500])
    # create feature vectors for reviews
    vectorized_reviews = []
    for review in train_cleaned_reviews:
        vec = []
        # compute the % of words in cluster 0, 1, 2, ... 500
        print("computing embeddings and clusts")
        embeddings_for_words =  review_to_embedding(final_embeddings, word_to_idx_dict, review)
        preds = kmeans.predict(embeddings_for_words) # cluster assigments for embeddings
        assert(len(embeddings_for_words) == len(preds))
        counts = Counter(preds)
        print("creating feature vector for review")
        for i in num_clusts:
            prop = counts[i] / float(len(preds)) # what proportion of words were assigned to clust i?
            vec.append(prop)
        vectorized_reviews.append(vec)
    vectorized_reviews = np.array(vectorized_reviews)
    print("done creating feature vectors for reviews")
    print(vectorized_reviews.shape)
    print(y.shape)
