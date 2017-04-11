from collections import defaultdict, Counter
import numpy as np
from queue import Queue, PriorityQueue

def make_dict_from_data(data_file):
	with open(data_file, 'r') as f:
		data = f.readlines()
	vocab = defaultdict(int)
	for line in data:
		words = line.strip().split(" ") #strip removes trailing and leading whitespace, tabs, newlines, etc
		# the " " param is optional here
		for word in words:
			vocab[word]+=1
	return vocab
















if __name__ == '__main__':
	data_file = 'data/input.txt'
	vocab = make_dict_from_data(data_file)
	vocab = Counter(vocab)
	common = vocab.most_common()
	print(len(common))
	print(len(vocab))