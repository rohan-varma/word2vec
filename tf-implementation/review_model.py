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
    """Takes a word, looks it up in an index dictionary, and returns the word's
    embedding."""
    try:
        idx = word_to_idx_dict[word]
    except KeyError:
        print("EMBEDDING LOOKUP FAILED for word: {}, defaulting to last embedding".format(word))
        return embeddings[-1]
    ret = embeddings[idx]
    assert ret.shape[0] == 128, "Houston we've got a problem"
    return ret

def review_to_embedding(embeddings, word_to_idx_dict, review):
    """Converts a review to it's (word-level) embedding"""
    words = review.split()
    embedding_matrix = [embeddings_lookup(embeddings, word_to_idx_dict, word)
                        for word in words]
    embedding_matrix = np.array(embedding_matrix)
    return embedding_matrix

def gen_batch(x_train, y_train, batch_size):
    """Generates a random batch of data for learning."""
    assert x_train.shape[0] == y_train.shape[0] # should have same dim
    assert batch_size <= x_train.shape[0] # batches can't be bigger than data
    merged = np.zeros((x_train.shape[0], x_train.shape[1] + y_train.shape[1]))
    merged[:, :-y_train.shape[1]] = x_train
    merged[:, -y_train.shape[1]:] = y_train
    np.random.shuffle(merged)
    x_train = merged[:, :-y_train.shape[1]]
    y_train = merged[:, -y_train.shape[1]:]
    batch = merged[np.random.choice(merged.shape[0], batch_size, replace=False), :]
    x_batch = batch[:, :-y_train.shape[1]]
    y_batch = batch[:, -y_train.shape[1]:]
    assert x_batch.shape[0] == y_batch.shape[0] == batch_size
    assert x_batch.shape[1] == x_train.shape[1]
    assert y_batch.shape[1] == y_train.shape[1]
    return x_batch, y_batch


def main():
    # the following code just reads in the word embeddings and a mapping from word to vector.
    try:
        print("loading embeddings")
        with open('data/embeddings.txt', 'rb') as f:
            final_embeddings, reverse_dictionary = pickle.load(f)
    except IOError:
        print("please run word2vec.py which will learn and save the embeddings.")
    try:
        print("loading data")
        with open('data/data.txt', 'rb') as f:
            train_cleaned_reviews, test_cleaned_reviews, vocab, y = pickle.load(f)
            print(type(y))
            print("done loading")
    except IOError:
        print("please run word2vec.py which will load and save the data")

    # variables needed later
    num_clusts = 500
    word_to_idx_dict = {v: k for k, v in reverse_dictionary.items()}

    # arguments to the script that load already processed data and run checks on them.
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="load clusters from file data/kmeans.pkl",
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
        kmeans = joblib.load('data/kmeans.pkl')
        li = joblib.load('data/reviews_and_labels.txt')
        assert(len(li) == 2)
        vectorized_reviews, y = li[0], li[1]
    else:
        print("clustering")
        kmeans = KMeans(n_clusters = 500, max_iter = 10000).fit(final_embeddings)
        print("done clustering, writing cluster object to data file kmeans.pkl")
        joblib.dump(kmeans, 'data/kmeans.pkl')
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
    joblib.dump([vectorized_reviews, y], "data/reviews_and_labels.txt")
    print("creating model")
    # convert if necessary
    labels = y.as_matrix() if not isinstance(y, np.ndarray) else y
    assert(labels.all() == y.all())
    from sklearn.model_selection import train_test_split
    y = y.reshape((y.shape[0], 1))
    vectorized_reviews = vectorized_reviews.astype(np.float32)
    y = y.astype(np.float32)
    print("creating model")
    from keras.utils.np_utils import to_categorical
    y = to_categorical(y)
    vectorized_reviews_train, vectorized_reviews_test, y_train, y_test = train_test_split(vectorized_reviews, y)
    print(vectorized_reviews.shape)
    print(y.shape)
    print(y_train.shape)
    print(y_test.shape)
    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape = shape))

    lr = 0.1
    neurons = 50

    x = tf.placeholder(tf.float32, shape = [None, vectorized_reviews.shape[1]])
    y_ = tf.placeholder(tf.float32, shape = [None, 2])
    W_1, b_1 = weight([vectorized_reviews.shape[1], neurons]), bias([neurons])
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    W_2, b_2 = weight([neurons, 30]), bias([30])
    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)
    W_3, b_3 = weight([30, 2]), bias([2])
    y = tf.matmul(h_2, W_3) + b_3

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
    opt_step = tf.train.MomentumOptimizer(lr, momentum = 0.5,
                                          use_nesterov=True).minimize(cross_entropy_loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    acc_file = open('cluster-acc.txt', 'w')
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            x_batch, y_batch = gen_batch(vectorized_reviews_train, y_train, 128)
            opt_step.run(feed_dict = {x: x_batch, y_: y_batch})
            if i % 500 == 0:
                acc = accuracy.eval(feed_dict = {x: vectorized_reviews_train,
                                                 y_: y_train})
                loss = cross_entropy_loss.eval(feed_dict = {x: vectorized_reviews_train, y_: y_train})
                print("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))
                acc_file.write("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))
        test_acc = accuracy.eval(feed_dict = {x: vectorized_reviews_test, y_: y_test})
        print("test acc: {}".format(test_acc))
        acc_file.write("test acc: {}".format(test_acc))
        acc_file.close()



if __name__ == '__main__':
    main()
