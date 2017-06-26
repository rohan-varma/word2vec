import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

def generate_review_features(review, word_to_idx_dict, embeddings):
    x = []
    for word in review.split():
        try:
            embedding = embeddings[word_to_idx_dict[word]]
        except KeyError:
            print("KeyError: {} not in embeddings, using last embedding as default".format(word))
            last_word = list(word_to_idx_dict.keys())[-1]
            embedding = embeddings[word_to_idx_dict[last_word]]
        x.append(embedding)
    return np.array(x)

def generate_batch(features_train_clean_review, y, batch_size = 64):
    """Generates a random batch of batch_size from the dataset
    Params:
    features_train_clean_review: the word vector features for the reviews
    y: the labels
    batch_size: the size of batch to generate (must be <= the dataset size)
    """
    x_batch, y_batch = [], []
    for _ in range(batch_size):
        randint = np.random.randint(0, features_train_clean_review.shape[0])
        features, label = features_train_clean_review[randint], y[randint]
        x_batch.append(features)
        y_batch.append(label)
    x_batch, y_batch = np.array(x_batch), np.array(y_batch)
    return x_batch, y_batch



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

    # a dictionary that maps words to indices in the word vector array.
    word_to_idx_dict = {v: k for k, v in reverse_dictionary.items()}
    # generate the features (ie, array of word vectors) for each review.
    features_train_clean_review = [generate_review_features(rev, word_to_idx_dict, final_embeddings) for rev in train_cleaned_reviews]
    features_train_clean_review = np.array(features_train_clean_review)
    y = y.values.reshape((y.shape[0], 1))

    print("generating a sample batch")
    x_batch, y_batch = generate_batch(features_train_clean_review, y, batch_size = 10)
    print("{} {}".format(x_batch.shape, x_batch[0].shape))
    print("{} {}".format(y_batch.shape, y_batch[0].shape))
    print("done generating batch")
    exit()
    # variables for the rnn
    RNNconfig = {
    'num_steps' : 5, # higher n = capture longer term dependencies, but more expensive (and potential vanishing gradient issues)
    'batch_size' : 200,
    'state_size' :10,
    'learning_rate' : 0.1
    }

    num_classes = 2
    x = tf.placeholder(tf.int32, [RNNconfig['batch_size'], RNNconfig['num_steps']], name='input_placeholder')
    y = tf.placeholder(tf.int32, [RNNconfig['batch_size'], RNNconfig['num_steps']], name='labels_placeholder')
    init_state = tf.zeros([RNNconfig['batch_size'], RNNconfig['state_size']])
    # Turn our x placeholder into a list of one-hot
    # rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
    x_one_hot = tf.one_hot(x, num_classes) # note: num_classes is not an RNN variable...
    rnn_inputs = tf.unstack(x_one_hot, axis=1)

    cell = tf.contrib.rnn.BasicRNNCell(RNNconfig['state_size'])
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state = init_state)
    # the longer version of what we just did above
    # with tf.variable_scope('rnn_cell'):
    #     W = tf.get_variable('W', [num_classes + RNNconfig['state_size'], RNNconfig['state_size']])
    #     b = tf.get_variable('b', [RNNconfig['state_size']], initializer=tf.constant_initializer(0.0))
    #
    # def rnn_cell(rnn_input, state):
    #     with tf.variable_scope('rnn_cell', reuse=True):
    #         W = tf.get_variable('W', [num_classes + RNNconfig['state_size'], RNNconfig['state_size']])
    #         b = tf.get_variable('b', [RNNconfig['state_size']], initializer=tf.constant_initializer(0.0))
    #     return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [RNNconfig['state_size'], num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
    y_as_list = tf.unstack(y, num=RNNconfig['num_steps'], axis=1)

#losses and train_step
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
            logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(total_loss)

    def train_network(num_epochs, verbose=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            # batch = gen_batch(features_train_clean_review)
            # for idx, epoch in enumerate(gen_epochs(num_epochs, RNNconfig['num_steps'])):
            #     training_loss = 0
            #     training_state = np.zeros((RNNconfig['batch_size'], RNNconfig['state_size']))
            #     if verbose:
            #         print("\nEPOCH", idx)
            #     for step, (X, Y) in enumerate(epoch):
            #         tr_losses, training_loss_, training_state, _ = \
            #             sess.run([losses,
            #                       total_loss,
            #                       final_state,
            #                       train_step],
            #                           feed_dict={x:X, y:Y, init_state:training_state})
            #         training_loss += training_loss_
            #         if step % 100 == 0 and step > 0:
            #             if verbose:
            #                 print("Average loss at step", step,
            #                       "for last 100 steps:", training_loss/100)
            #             training_losses.append(training_loss/100)
            #             training_loss = 0

        return training_losses
