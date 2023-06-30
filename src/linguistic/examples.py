import semantic_vector as sv
import sentimental_analysis as sa

import time
from nltk.corpus import stopwords
from nltk.corpus.reader import BracketParseCorpusReader, PlaintextCorpusReader

from nltk.util import bigrams
from nltk.lm import Vocabulary, MLE
from nltk.lm.preprocessing import pad_both_ends, flatten

from gensim.models import Word2Vec
import gensim.downloader



if __name__ == "__main__":

### SENTIMENTAL ANALYSIS ###:
	# train_corpus = BracketParseCorpusReader(root="corpora", fileids=["train.txt"])
	# test_corpus = BracketParseCorpusReader(root="corpora", fileids=["test.txt"])
	# support = 3

	# # Tokenizing corpus
	# train_tk, train_sents_tkd, train_tk_freq = sa.tokenize(train_corpus, support, True)

	# # Tokenized vocabulary
	# vocab = Vocabulary(train_sents_tkd, unk_cutoff=support)

	# # Padding flat tokenized sentences
	# train_tkd_padflat = list(flatten(pad_both_ends(sent, n=2) for sent in train_sents_tkd))

	# ### N-GRAM: 2
	# # N-gram, flat padded tokenized sentences
	# train_tkd_padflat_bi = list(bigrams(train_tkd_padflat))
	# # N-gram and Padding, padded tokenized sentences
	# train_tkd_pad_bi = []
	# for i in range(len(train_sents_tkd)):
	#  	train_tkd_pad_bi.append(list(bigrams(pad_both_ends(train_sents_tkd[i], n=2)))) # necessary bi-gram???

	# # Maximum Likelyhood Estimation (by nltk class)
	# # check: https://www.nltk.org/api/nltk.lm.html?highlight=vocabulary#module-nltk.lm.vocabulary
	# lm = MLE(2)
	# lm.fit(train_tkd_pad_bi, vocab)
	# word_prob = sa.nextword_prob("<s>", lm)

	# # Laplace Smoothening (by nltk class)
	# # lpc = Laplace(2)
	# # lpc.fit(train_tk_pad_bi, vocab)

	# # built (by class built above)

	# mlm = sa.My_language_model(train_tkd_pad_bi, train_tkd_pad_bi, vocab) # flat_pad, n, support
	# mlm.nxt_words("<s>")
	# mlt.perplexity(test)

### SEMANTIC VECTOR ####:
	corpus = PlaintextCorpusReader(root="corpora", fileids=["corpus.txt"])
	raw_words, raw_sents, raw_paras = corpus.words(), corpus.sents(), corpus.paras()
	# Corpus processing
	corpus_words = sv.rmv_punctuation(raw_words)
	corpus_sents = [sv.rmv_punctuation(sent) for sent in raw_sents]

	# Task 2.1 - TF_IDF
	tf_idf = sv.tfidf_weight_matrix(corpus_words)
	# Task 2.2 - PPMI
	ppmi = sv.ppmi_weight_matrix(corpus_words, 1e-4)
	# Task 2.3 Word2Vec
	"""
	1. Using a word embedding with 100 dimensions --> vector_size (int, optional) – Dimensionality of the word vectors.
	2. Using a window size of 5 words --> window (int, optional) – Maximum distance between the current and predicted word within a sentence.
	3. Accepting any word occurring in the preprocessed worpus at least twice --> min_count (int, optional) – Ignores all words with total frequency lower than this.
	4. Using skipgrams rather than continuous bag of words --> sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
	5. Generating 10 negative samples per word --> negative (int, optional) – Negative sampling will be used. how many “noise words” (usually between 5-20).
	6. Performing 300 iterations of the underlying gradient descent optimization --> epochs (int, optional) – Number of iterations (epochs) over the corpus.
	"""
	# model = Word2Vec(vector_size=100, window=2, min_count=2, workers=1, sg=1, negative=10, epochs=300)
	# model.build_vocab(corpus_sents)  # prepare the model vocabulary
	# model.train(corpus_sents, total_examples=model.corpus_count, epochs=model.epochs)
	# model.train([["hello", "world"]], total_examples=1, epochs=1)

	for word in ['sometimes', 'relief', 'took']:
		dist = sv.semantic_dist(5, word, tf_idf)
		print("The TF-IDF 5 closest words for '{}', are: {}".format(word, [t[0] for t in dist ]))
		dist = sv.semantic_dist(5, word, ppmi)
		print("The PPMI  5 closest words for '{}', are: {}".format(word, [t[0] for t in dist ]))
		# sims = model.wv.most_similar(word, topn=5)
	# 	print("The W2V 5 closest words for '{}', are: {}".format(word, [t[0] for t in sims]))
	# 	# vector = model.wv[word]  					   							# get numpy vector of a word
	# 	# model.most_similar(positive=['woman', 'king'], negative=[], topn=1)	# vector operation
	# 	# model.doesnt_match("breakfast cereal dinner lunch".split())
	# 	# model.similarity('woman', 'man')										# scalar result

	# # Pre-trained model
	# google_news_300 = gensim.downloader.load('word2vec-google-news-300')
	# #upload the word2vec-google-news-300 pre-trained embedding
	# google_news_300 = gensim.downloader.load('word2vec-google-news-300')

	# ggl = google_news_300.most_similar('investigation', topn=5)
	# print("The Word2Vec  5 closest words for '{}', are: {}".format('investigation', [t[0] for t in ggl ]))

