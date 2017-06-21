### Word2Vec
Word2Vec implementation in Python and using tensorflow, 
with application to learning word vectors on an IMDB
movie-reviews dataset. 

Investigates several models, including a bag of words model, a neural network that learns from clusters of word-vectors, and an RNN whose inputs are word-vectors. 

### Usage

`make`: Will run all of the models (as well as output accuracy reports), download the necessary data, and write the preprocessed data to persistent storage so future runs don't have to do preprocessing/redownloading.
	
`make` targets: 
	
- `bag-of-words`: runs the bag of words model and writes an accuracy report to file `bag-of-words/bag-of-words-acc.txt`
		
- `word-vectors`: runs a Tensorflow model to learn word vectors from the review data, and outputs a picture of a TSNE word vector visualization to file `tf-implementation/TSNE.png`
		
- `cluster-model`: runs a feature engineering algorithm that generates features based on the clustering of a review's word vectors (ie, the frequency of each word vectors in a particular cluster in the review) followed by a Tensorflow neural network model whose inputs are these clusters. An accuracy report is written to file `tf-implementation/cluster-acc.txt`.
		
- `rnn-model`: runs a Tensorflow RNN model who's inputs are the learned word vectors made by the `word-vectors` target. An accuracy report is written to file `tf-implementation/rnn-acc.txt`. 


