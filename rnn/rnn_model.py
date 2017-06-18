import numpy as np
import tensorflow as tf
import pickle

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

def generate_batch(features_train_clean_review, batch_size = 64):
    batch = []
    for i in range(batch_size):
        idx = np.random.randint(0, features_train_clean_review.shape[0])
        batch.append(features_train_clean_review[idx])
    return np.array(batch)



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
    word_to_idx_dict = {v: k for k, v in reverse_dictionary.items()}
    features_train_clean_review = [generate_review_features(rev, word_to_idx_dict, final_embeddings) for rev in train_cleaned_reviews]
    features_train_clean_review = np.array(features_train_clean_review)
    print(features_train_clean_review[0].shape)
    batch = generate_batch(features_train_clean_review, batch_size = 10)
    print(batch.shape[0])
    print(batch[0].shape)
