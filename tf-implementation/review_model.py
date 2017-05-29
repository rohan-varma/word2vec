import numpy as np
import tensorflow as tf
import pickle
import sklearn
from sklearn.cluster import KMeans
import tensorflow as tf
from collections import Counter
import argparse
from sklearn.externals import joblib # for dumping models
# probably should not be doing this...
import warnings
warnings.filterwarnings("ignore")

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def embeddings_lookup(embeddings, word_to_idx_dict, word):
    try:
        idx = word_to_idx_dict[word]
    except KeyError:
        print("EMBEDDING LOOKUP FAILED for word: {}".format(word))
        return word
    ret = embeddings[idx]
    assert ret.shape[0] == 128, "Houston we've got a problem"
    return ret

def review_to_embedding(embeddings, word_to_idx_dict, review):
    words = review.split()
    embedding_matrix = [embeddings_lookup(embeddings, word_to_idx_dict, word)
                        for word in words]
    embedding_matrix = np.array(embedding_matrix)
    # print(embedding_matrix.shape)
    # exit()
    return embedding_matrix


if __name__ == '__main__':
    # the following code just reads in the word embeddings and a mapping from word to vector.
    try:
        print("loading embeddings")
        with open('embeddings.txt', 'rb') as f:
            final_embeddings, reverse_dictionary = pickle.load(f)
    except IOError:
        print("please run word2vec.py which will learn and save the embeddings.")
    try:
        print("loading data")
        with open('../data.txt', 'rb') as f:
            train_cleaned_reviews, test_cleaned_reviews, vocab, y = pickle.load(f)
            print(type(y))
            print("done loading")
    except IOError:
        print("please run word2vec.py which will load and save the data")

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
    # clustering embeddings - but check if written  in a file
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="load clusters from file cluster.data",
                        action = "store_true")
    parser.add_argument("-t", "--test", help = "run sanity checks on data", action = "store_true")
    args = parser.parse_args()
    if args.test:
        print("Sanity checking: every word in vocab should be in the dict")
        for v in list(vocab):
            if v not in list(word_to_idx_dict.keys()) : print("SANITY ERROR: word {} not in dict.".format(v))

        print("Sanity checking: every word in every cleaned review should be in the dict")
        for review in train_cleaned_reviews:
            words = review.split()
            for word in words:
                if word not in list(word_to_idx_dict.keys()) : print("SANITY ERROR: word {} in some review not in dict".format(word))
        for review in test_cleaned_reviews:
            words = review.split()
            for word in words:
                if word not in list(word_to_idx_dict.keys()) : print("SANITY ERROR: word {} in some test review not in dict".format(word))

    if args.load:
        print("loading cluster object from file kmeans.pkl")
        kmeans = joblib.load('kmeans.pkl')
    else:
        print("clustering")
        kmeans = KMeans(n_clusters = 500, max_iter = 10000).fit(final_embeddings)
        print("done clustering, writing cluster object to data file kmeans.pkl")
        joblib.dump(kmeans, 'kmeans.pkl')
    # assign embeddings to clusters
    print("embedding 500 prediction: {}".format(kmeans.predict(final_embeddings[500])))
    # create feature vectors for reviews
    m = 0
    vectorized_reviews = []
    for review in train_cleaned_reviews:
        m+=1
        vec = []
        # compute the % of words in cluster 0, 1, 2, ... 500
        if m % 100 == 0 : print("iteration {}, computing embeddings and clusts".format(m))
        # the following code gets an embedding matrix for a review.
        embeddings_for_words =  review_to_embedding(final_embeddings, word_to_idx_dict, review)

        if m % 100 == 0 : print("iteration {}, (review_len * embedding len) : {}".format(m,embeddings_for_words.shape))
        preds = []
        for i in range(embeddings_for_words.shape[0]):
            preds.append(kmeans.predict(embeddings_for_words[i])[0])

        counts = Counter(list(preds))
        if m % 100 == 0 : print("iteration {}, creating feature vector for review\n".format(m))
        # now, we create a feature vector for the review where the ith featur denotes the proportion of word embeddings assigned to that cluster
        for i in range(num_clusts):
            prop = counts[i] / float(len(preds)) # what proportion of words were assigned to clust i?
            vec.append(prop)
        vectorized_reviews.append(vec)
    vectorized_reviews = np.array(vectorized_reviews)
    print("done creating feature vectors for reviews")
    print(vectorized_reviews.shape)
    print(y.shape)
    print("creating model")
    exit()
    lr = 0.1
    hidden_layer = 50
    num_iters = 5000
    labels = y.as_matrix() # turn labels into np array
    x = tf.placeholder(tf.float32, shape = [None, vectorized_reviews.shape[1]])
    y = tf.placeholder(tf.float32, shape = [None, 1]) # binary outputs

    W_fc1 = weight_variable([vectorized_reviews.shape[1], hidden_layer])
    b_fc1 = bias_variable([hidden_layer])
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = weight_variable([hidden_layer, 1])
    b_fc2 = weight_variable([1])
    y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits = y_out)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, axis = 1), tf.argmax(y_out, axis = 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    print("running model")
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1):
            pass # TODO generate batch
            sess.run(optimizer, feed_dict = {x: vectorized_reviews, y: labels})
            xent = cross_entropy.eval(feed_dict = {x: vectorized_reviews, y: labels})
            print("epoch: {}, loss: {}".format(i, xent))
