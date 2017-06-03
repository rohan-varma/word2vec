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
        print("EMBEDDING LOOKUP FAILED for word: {}, defaulting to last embedding".format(word))
        return embeddings[-1]
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
        with open('./data.txt', 'rb') as f:
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
        li = joblib.load('reviews_and_labels.txt')
        assert(len(li) == 2)
        vectorized_reviews, y = li[0], li[1]
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
    joblib.dump([vectorized_reviews, y], "reviews_and_labels.txt")
    print("creating model")
    labels = y.as_matrix() if not isinstance(y, np.ndarray) else y
    assert(labels.all() == y.all())
    from sklearn.model_selection import train_test_split
    y = y.reshape((y.shape[0], 1))
    vectorized_reviews_train, vectorized_reviews_test, y_train, y_test = train_test_split(vectorized_reviews, y)
    print("creating model")
    # keras test
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.utils.np_utils import to_categorical
    y = to_categorical(y)
    vectorized_reviews_train, vectorized_reviews_test, y_train, y_test = train_test_split(vectorized_reviews, y)
    model = Sequential()
    model.add(Dense(units=300, input_dim=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.fit(vectorized_reviews_train, y_train, epochs=5000, batch_size=128)
    print()
    loss_and_metrics = model.evaluate(vectorized_reviews_test, y_test, batch_size=128)
    print(loss_and_metrics)
    exit()
    x = tf.placeholder(tf.float32, shape=[None, 500])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def weight_variable(shape):
        """Initializes weights randomly from a normal distribution
        Params: shape: list of dimensionality of tensor
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Initializes the bias term randomly from a normal distribution.
        Params: shape: list of dimensionality for the bias term.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def fc_layer(scope, x, weight_shape, activation = 'relu', keep_prob = 1.0):
        with tf.variable_scope(scope):
            W_fc = weight_variable(weight_shape)
            b_shape = [weight_shape[-1]]
            b_fc = bias_variable(b_shape)
            h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, keep_prob=keep_prob)
            return h_fc_drop

    # create weights and biases and function for our first layer
    W_fc1, b_fc1 = weight_variable([500, 100]), bias_variable([100])
    # hidden layer computes relu(Wx + b)
    #h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    keep_prob_1 = tf.placeholder(tf.float32)
    # add dropout: discard activations with probability given by keep_prob
    #h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob_1)

    h_fc1_dropout = fc_layer("layer-1", x, [vectorized_reviews.shape[1], 100], activation = 'relu',
                             keep_prob = keep_prob_1)

    # create w, b, and function for our next layer
    W_fc2, b_fc2 = weight_variable([100, 30]), bias_variable([30])
    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

    # # add dropout
    keep_prob_2 = tf.placeholder(tf.float32)

    # # discard second hidden layer activations with keep_prob_2 probability
    #h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob_2)
    h_fc2_dropout = fc_layer("layer-2", h_fc1_dropout, [100, 30], activation = 'relu',
                             keep_prob = keep_prob_2)
    # define w and b for the softmax layer
    W_fc3, b_fc3 = weight_variable([30, 1]), bias_variable([1])

    # softmax Output
    y_pred = tf.nn.softmax(tf.matmul(h_fc2_dropout, W_fc3) + b_fc3)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_pred))
    sae = -tf.reduce_sum(y_-y_pred)
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #sae_train_step = tf.train.AdamOptimizer(1e-4).minimize(sae)
    train_step = tf.train.MomentumOptimizer(1e-4, 0.5, name='Momentum', use_nesterov=True).minimize(cross_entropy)
    # accuracy variables
    cp = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(cp, tf.float32))
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            # iterate for 10k epochs and run batch SGD.
            sess.run(train_step, feed_dict={x: vectorized_reviews_train, y_: y_train, keep_prob_1: 0.8,
                                            keep_prob_2: 0.5})
            if i % 100 == 0:
                print("epoch: {}".format(i + 1))
                print(acc.eval(feed_dict={x: vectorized_reviews_test, y_: y_test, keep_prob_1: 1.0,
                                          keep_prob_2: 1.0}))
        print("done training!")
        test_acc = acc.eval(feed_dict={x: vectorized_reviews_test, y_: y_test, keep_prob_1: 1.0,
                                       keep_prob_2: 1.0})
        print("test acc: {}".format(test_acc))

    sess.close()
