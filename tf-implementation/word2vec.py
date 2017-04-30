import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from bs4 import BeautifulSoup
sys.path.append('../')
from utils import *
import pandas as pd
import re
import collections
from nltk.corpus import stopwords
import random

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
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
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

def create_model():
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                                                embedding_size], -1.0, 1.0))
    # "Xavier" init
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    train_inputs = tf.placeholder(tf.float32, shape = [None, vocab_size])
    train_labels = tf.placeholder(tf.float32, shape = [None, 1])



if __name__ == '__main__':
    clean_reviews, y = extract_and_clean_data('../data/labeledTrainData.tsv')
    vocab = build_vocab(clean_reviews, keep_dups = True)
    words, vocabulary_size = vocab, len(vocab)
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    del words  # Hint to the OS to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    print(batch.shape)
    print(labels.shape)