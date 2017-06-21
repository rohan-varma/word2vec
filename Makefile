.PHONY: default bag-of-words word-vectors
.SILENT:
default: bag-of-words word-vectors cluster-model rnn-model


bag-of-words:
	cd bag-of-words; \
	if [[ ! -f data.txt ]]; then \
		echo "bag-of-words: running w/o --load because file data.txt does not exist...expect this to take a while"; \
		python3 bwmodel.py; \
	else \
		echo "bag-of-words: running with --load, loading preprocessed data from file data.txt"; \
		python3 bwmodel.py --load; \
	fi; \
	cd ..;

word-vectors:
	cd tf-implementation; \
	if [[ ! -f data/data.pik ]]; then \
		echo "word-vectors: running without --load"; \
		python3 word2vec.py; \
	else \
		echo "word-vectors: running with --load, loading preprocessed data from file data/data.pik"; \
		python3 word2vec.py --load; \
	fi; \
	cd ..;

cluster-model:
	cd tf-implementation; \
	C1=(-f data/kmeans.pkl); \
	C2=(-f data/reviews_and_labels.txt); \
	if [[ ! $C1 || ! $C2 ]]; then \
		echo "cluster-model: running without --load"; \
		python3 review_model.py; \
	else \
		echo "cluster-model: running with --load, loading stored clusters from data/kmeans.pkl and preprocessed data from data/reviews_and_labels.txt"; \
		python3 review_model.py --load; \
	fi; \
	cd ..;

rnn-model:
	echo "rnn-model: TODO"
