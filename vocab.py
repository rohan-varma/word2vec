from collections import defaultdict, Counter
import numpy as np
from nltk.corpus import stopwords

def make_dict_from_data(data_file):
	with open(data_file, 'r') as f:
		data = f.readlines()
	vocab = defaultdict(int)
	for line in data:
		words = line.strip().split() #strip removes trailing and leading whitespace, tabs, newlines, etc
		# the " " param is optional here
		for word in words:
			vocab[word]+=1
	return vocab

def encode_word(word, vocab):
	vocab_words = list(vocab.keys())
	assert(word in vocab_words)
	a = np.zeros(shape = len(vocab_words))
	idx = vocab_words.index(word)
	a[idx] = 1
	return a

def encode_all_words(vocab):
	vocab_words = list(vocab.keys())
	encodings = []
	for word in vocab_words:
		encodings.append(encode_word(word, vocab))
	return encodings


















if __name__ == '__main__':
	data_file = 'data/input.txt'
	vocab = make_dict_from_data(data_file)
	vocab = Counter(vocab)
	print(len(vocab.keys()))
	common = vocab.most_common(100)
	stop = stopwords.words("english")
	common, stop = set(common), set(stop)
	for k in list(vocab.keys()):
		if k in stop or k in common:
			try:
				del vocab[k]
			except KeyError:
				pass
	encoded_words = encode_all_words(vocab)
	print(encoded_words.shape)

