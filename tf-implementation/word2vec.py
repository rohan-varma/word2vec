import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from bs4 import BeautifulSoup
sys.path.append('../')
import pandas as pd
import re
import collections
from nltk.corpus import stopwords
import random
import pickle
import argparse
import math

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

def extract_and_clean_data(path_to_file):
    """Takes in a path to the reviews dataset and returns cleaned reviews along
    with their sentiments (binary labels)
    Params:
    path_to_file: the path to the data file
    Returns:
    clean_reviews: The cleaned reviews (punctation and common words stripped)
    y: the labels (sentiments for each review)
    """
    data = pd.read_csv(path_to_file, header = 0, delimiter='\t', quoting=3)
    clean_reviews = [clean(data['review'][i])
                     for i in range(len(data['review']))]
    y = data['sentiment']
    return clean_reviews, y

def build_vocab(cleaned_reviews, keep_dups = False):
    """Takes in a bunch of reviews and returns a list V of all the words
    (no dups)
    Params: cleaned_reviews
    Returns: vocab: a list of all of the words
    """
    li = []
    for review in cleaned_reviews:
        a = review.split()
        for item in a:
            li.append(item)
    return list(set(li)) if not keep_dups else li


def word_one_hot(word, vocab):
    idx = vocab.index(word)
    if(idx < 0):
        raise ValueError("vocab must contain word")
    v = np.zeros(len(vocab))
    v[idx] = 1
    return np.array(v)

def create_training_examples_one_hot(review, vocab, window_size = 1):
    """Creates the word pairs for the skip-gram model.
    Params: review - a single (cleaned) review
    window_size: how many context words to use.
    Default = 1 - ie, use the word to the left and to the right as the context
    Returns:
    data - a bunch of training data
    """
    words = review.split()
    data, labels = [], []
    for i in range(len(words)):
        left = [words[i-j] for j in range(1, window_size + 1) if i-j >= 0]
        right = [words[i+j] for j in range(1, window_size + 1) if i+j < len(words)]
        neighbors = left + right
        for item in neighbors:
            data.append(word_one_hot(words[i], vocab))
            labels.append(word_one_hot(item, vocab))
    return data, labels

def build_dataset(words, vocabulary_size):
    # create a counter for the msot common words
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  # get word indices
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

    # Visualize
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename)
    return 1
     # plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="load data structures from a file",
                        action = "store_true")
    args = parser.parse_args()
    if args.load:
        print("loading data from file data.pik")
        with open('data.pik', 'rb') as f:
            clean_reviews, y, data, count, dictionary, reverse_dictionary, vocabulary_size = pickle.load(f)
    else:
        clean_reviews, y = extract_and_clean_data('../data/labeledTrainData.tsv')
        vocab = build_vocab(clean_reviews, keep_dups = False)
        words, vocabulary_size = vocab, len(vocab)
        data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)

        del words
        with open('data.pik', 'wb') as f:
            pickle.dump([clean_reviews, y, data, count, dictionary,
                         reverse_dictionary, vocabulary_size], f, -1)

    print("cleaned reviews len: {}".format(len(clean_reviews)))
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    data_index = 0
    batch_size = 8
    batch, labels = generate_batch(batch_size=batch_size, num_skips=2, skip_window=1)
    print(batch.shape)
    print(labels.shape)
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_size = 16
    embedding_size = 128
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        # creates a vocab_size * embedding_size matrix. For the ith word in the vocab,
    # the vector embeddings[i] is the corresponding embedding.
    print(vocabulary_size)
    print(embedding_size)
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # treats the embedding matrix as a lookup table so we can get the embeddings

    # "Xavier" init
    nce_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size],
        stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    num_sampled = 64
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss_fun = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_fun)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    nsteps = 500000
    # exit()
    with tf.Session() as sess:
        sess.run(init)
        avg_loss = 0.
        for step in range(nsteps):
            batch_inputs, batch_labels = generate_batch(batch_size,
                                                        num_skips, skip_window) # 2 = num skips, 1 = skip window
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss = sess.run([optimizer, loss_fun], feed_dict = feed_dict)
            avg_loss+=loss
            if step % 10000 == 0:
                if step != 0: avg_loss /= 10000
                print("Average loss at epoch {}: {}".format(step, avg_loss))
                avg_loss = 0.
        final_embeddings = normalized_embeddings.eval()
    try:
         from sklearn.manifold import TSNE
         import matplotlib.pyplot as plt
         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
         plot_only = 500
         low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
         labels = [reverse_dictionary[i] for i in range(plot_only)]
         plot_with_labels(low_dim_embs, labels)
    except ImportError:
        print("please install sklearn, matplotlib, and scipy for tsne embeddings")
    print(final_embeddings.shape)
    print(len(reverse_dictionary.values()))
    with open('embeddings.txt', 'wb') as f:
        print("dumping embeddings")
        pickle.dump([final_embeddings, reverse_dictionary], f, -1)
    with open('data.txt', 'wb') as f:
        print("dumping reviews, vocab, and labels")
        pickle.dump([train_cleaned_reviews, test_cleaned_reviews, vocab, y], f, -1)
