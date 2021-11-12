"""Semantic Vectors
* :function:`.tokenize`
* :class:`.laplace`
* :function:`.bayesian`
* :function:`.Perplexity`
"""

from collections import Counter
import numpy as np
import num2words
import math
import pandas as pd
import time

import nltk
from nltk import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.corpus.reader import BracketParseCorpusReader, PlaintextCorpusReader

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.probability import FreqDist
from nltk.util import bigrams, ngrams

from nltk.lm import Vocabulary, MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten


def tokenize(corpus, support, verbose=False):
	""" Tokenize corpus according to a support (int) outputting:
	corpus_tokeninzed, corpus_sents_tokenized and tokens_freq dictionaire
	"""
	words, sents = corpus.words(), corpus.sents()

	# Corpus Vocabulary
	vocab = Vocabulary(words, unk_cutoff=support) # trying corpus instaed of words
	corpus_tkd = vocab.lookup(words) # list of tokens repect to vocab

	# Words percentage defined as '<UNK>'
	OOV = round(100*Counter(corpus_tkd)['<UNK>']/len(words),3)

	# Words frequency
	corpus_tks_freq = FreqDist(corpus_tkd)

	# Tokenizin sentences
	corpus_sents_tkd =[]
	for sent in sents:
		sent_tokenized = [vocab.lookup(word) for word in sent]
		corpus_sents_tkd.append(sent_tokenized)

	n=10
	test_bigram = list(bigrams(pad_both_ends(corpus_sents_tkd[n-1], n=2)))

	if verbose:
		print("The corpus total # of words is {}, of which {} are uniques.".format(len(words),len(set(words))))
		print("The corpus percentage of words OOV is {}.".format(OOV))
		print("The {}th tokenized sents bigrams is:{}.".format(n,test_bigram))
	return corpus_tkd, corpus_sents_tkd, corpus_tks_freq

class My_language_model:
	"""
	Creates language model based on bigram (only).
	For ngrams recussions: https://github.com/joshualoehr/ngram-language-model/blob/master/language_model.py
	input:
			- tks_pad   = MLE loaded with bigram(tk_pad) and vocab(tk_pad)
		 	- tks_bigrams_pad padded bi = vocab = lm.vocab #we can do it here
			- vocab       = token (by suppor) vocab

	ToDo:
	   -1. def _smooth(self)
		0. Runs entire dictionaire, waste of time. should it run by consult
	  	1. Put all basic variables at the __init__: tokens_pad
	  	2. Define all the inputs: n of grams, support of dictionaire
	  	3. support functions: _frequency dict.
	  	4. Decide is smoothed or not based on "n"
	  	5. Absorve/Adapt corpus preprocessing function
	"""
	def __init__(self, tks_pad, tks_bigrams_pad, vocab) -> None:
		self.tokens = tks_pad
		self.bigrams = tks_bigrams_pad
		self.vocab = vocab

		flat_tks_bigrams_pad = []
		for l in tks_bigrams_pad:
			flat_tks_bigrams_pad += l

		tks_freq = FreqDist(tks_pad)
		bigrams_freq = FreqDist(flat_tks_bigrams_pad)

		smoothed_prob = {} # {w:{h1=:1, h2=2},...} smoothed[a[b]]
		for word in vocab:
			prob = {}
			for ngram in bigrams_freq.keys():
				h, w = ngram[:-1], ngram[-1:]
				if word == w[0] and w[0] != '<UNK>' and h[0] != '<UNK>':
					prob[h[0]] = (bigrams_freq[h+w] + 1)/(tks_freq[h[0]] + len(vocab) -1) # (len(vocab)-1) because of <UNK>
			smoothed_prob[word] = prob

		self.smoothed_prob = smoothed_prob

	def score(self, w, hist):
		""" Returns the probability of a word for a given history"""

		if hist[0] in self.smoothed_prob[w].keys():
			return self.smoothed_prob[w][hist[0]]
		return

	def nxt_words(self, h, verbose=True):
		total = 0
		nextword_prob = {}
		for word in self.vocab:
			if word != h:
				if self.score(word, [h]):
					nextword_prob[word] = self.score(word, [h])
					total += self.score(word, [h])

		nextword_prob = sorted(nextword_prob.items(), key=lambda x: x[1], reverse=True)
		if verbose:
			print("The top3 most likely words after {}, are:{}".format(h, nextword_prob[:3]))
		return  nextword_prob

	def perplexity(self):
		pass

def nextword_prob(h, model, verbose=True):
	total = 0
	nextword_prob = {}

	for word in model.vocab:
		if word != h:
			if model.score(word, [h]):
				nextword_prob[word] = model.score(word, [h])
				total += model.score(word, [h])

	nextword_prob = sorted(nextword_prob.items(), key=lambda x: x[1], reverse=True)
	if verbose:
		print("The top3 most likely words after {}, are:{}".format(h, nextword_prob[:3]))
	return  nextword_prob

def rmv_punctuation(array):
	"Remove punctuation from a [sentence]"
	return [x.lower() for x in array if x.isalpha()]

def rmv_stopwords(array):
	"Remove 'stop words' from a [sentence]"
	return [x.lower() for x in array if not x in stopwords.words("english")]

def lemmatization(array):
	"Remove use lematization for words in [sentence]"
	lemma = WordNetLemmatizer()
	return [lemma.lemmatize(x, pos = "v") for x in [lemma.lemmatize(word, pos = "n") for word in array]]

def stemming(array):
	"Remove use steeming for words in [sentence]"
	pass
	return

def cosine_sim(v, w):
    cos_sim = np.dot(v, w)/(np.linalg.norm(v)*np.linalg.norm(w))
    return cos_sim

def semantic_dist(n, target, weight):
	""" """
	distance = {}
	for word in weight.keys():
		if word != target:
			w_dic = weight[word].copy()
			v_dic = weight[target].copy()

			# Making dictionaire at same size
			k2w = [k for k in v_dic if k not in w_dic]
			for k in k2w:
				w_dic[k] = 0
			k2v = [k for k in w_dic if k not in v_dic]
			for k in k2v:
				v_dic[k] = 0

			w_dic = sorted(w_dic.items(), key=lambda x: x[0])
			w = [t[1] for t in w_dic]
			v_dic = sorted(v_dic.items(), key=lambda x: x[0])
			v = [t[1] for t in v_dic]

			distance[word] = cosine_sim(v,w)

	sorted_dist = sorted(distance.items(), key=lambda x: x[1], reverse=True)
	return sorted_dist[:n]

def co_occurence_matrix(words):
	unique = set(words)
	InnerDict = {word:0 for word in unique}
	OutterDict, Dfj, N = {}, InnerDict.copy(), 0
	for i in range(len(words)):
		N += 1 # the total number of context window for all words
		if words[i] in InnerDict.keys():
			OutterDict[words[i]] = {} if words[i] not in OutterDict.keys() else OutterDict[words[i]]
			for j in [-2, -1, 1, 2]: # c_(i-2), c_(i-1), wi, c_(i-+1), c_(i+2)
					if 0 <= i + j < len(words) and words[i] != words[i+j]: # Cc_Matriz: cj not considered context of itself
						try:
							OutterDict[words[i]][words[i+j]] += 1
						except:
							OutterDict[words[i]][words[i+j]] = 1
					if 0 <= i + j < len(words): # Dfi: cj considered in context window of itself
						Dfj[words[i+j]] += 1
	return OutterDict, Dfj, N

def tfidf_weight_matrix(words):
	weights, Dfj, N = co_occurence_matrix(words)
	for w in weights.keys():
		for c in weights[w].keys():
			tf = math.log(weights[w][c]+1)	# Term Frequency smoothing (TF_ij)
			idf = math.log(N/Dfj[c])		# Inverse Document Frequency (IDF_j)
			weights[w][c] = tf * idf
	return weights

def mtx_smoothing(matrix, epsilon):
	""" Adding epsilon into all matrix entries."""
	total = 0
	for w in matrix.keys():
		for c in matrix[w].keys():
			matrix[w][c] += epsilon
			total += matrix[w][c]
	return matrix, total

def mtx_norm(matrix, total):
	""" Divide all matrix entries by the total sum."""
	for w in matrix.keys():
		for c in matrix[w].keys():
			matrix[w][c] = matrix[w][c]/total
	return matrix

def ppmi_weight_matrix (words, smoothing=None):

	weights, Dfj, N = co_occurence_matrix(words)

	# smoothingn
	epsilon = 1/len(Dfj)
	if smoothing:
		epsilon = smoothing

	# Smoothing entire weight matrix by epsilon
	e_weight, total = mtx_smoothing(weights, epsilon)

	# Normalized matrix
	p_weight = mtx_norm(e_weight, total)

	# Row sum (P_wi) & and columns sum (P_cj)
	P_w, P_c = {}, {}
	for k in p_weight:
		P_w[k] = sum(p_weight[k].values())
		P_c = {key: P_c.get(key, 0) + p_weight[k].get(key, 0) for key in set(P_c) | set(p_weight[k])}

	# Converting p_weight matrix into PPMI values
	for i in p_weight.keys():
		for j in p_weight[i].keys():
			pmi = math.log(p_weight[i][j]  / (P_w[i]*P_c[j]), 2)	# PMI
			p_weight[i][j] = max(pmi,0)							# PPMI

	return p_weight


### TAKS2 ####
if __name__ == "__main__":

	corpus = PlaintextCorpusReader(root="corpora", fileids=["corpus.txt"])
	raw_words, raw_sents, raw_paras = corpus.words(), corpus.sents(), corpus.paras()
	# Corpus processing
	corpus_words = rmv_punctuation(raw_words)

	# # Task 2.1
	# tf_idf = tfidf_weight_matrix(corpus_words)
	# for word in ['sometimes', 'relief', 'took']:
	# 	dist = semantic_dist(5, word, tf_idf)
	# 	print("The TF-IDF 5 closest words for '{}', are: {}".format(word, [t[0] for t in dist ]))

	# Task 2.2
	ppmi = ppmi_weight_matrix(corpus_words, 1e-4)
	for word in ['sometimes', 'relief', 'took']:
		dist = semantic_dist(5, word, ppmi)
		print("The PPMI  5 closest words for '{}', are: {}".format(word, [t[0] for t in dist ]))






# ### TASK 1 ###
# if __name__ == "__main__":

# 	train_corpus = BracketParseCorpusReader(root="corpora", fileids=["train.txt"])
# 	test_corpus = BracketParseCorpusReader(root="corpora", fileids=["test.txt"])
# 	support = 3

# 	# Tokenizing corpus
# 	train_tk, train_sents_tkd, train_tk_freq = tokenize(train_corpus, support, True)

# 	# Tokenized vocabulary
# 	vocab = Vocabulary(train_sents_tkd, unk_cutoff=support)

# 	# Padding flat tokenized sentences
# 	train_tkd_padflat = list(flatten(pad_both_ends(sent, n=2) for sent in train_sents_tkd))

# 	### N-GRAM: 2
# 	# N-gram, flat padded tokenized sentences
# 	train_tkd_padflat_bi = list(bigrams(train_tkd_padflat))
# 	# N-gram and Padding, padded tokenized sentences
# 	train_tkd_pad_bi = []
# 	for i in range(len(train_sents_tkd)):
# 	 	train_tkd_pad_bi.append(list(bigrams(pad_both_ends(train_sents_tkd[i], n=2)))) # necessary bi-gram???

# 	# Maximum Likelyhood Estimation (by nltk class)
# 	# check: https://www.nltk.org/api/nltk.lm.html?highlight=vocabulary#module-nltk.lm.vocabulary
# 	lm = MLE(2)
# 	lm.fit(train_tkd_pad_bi, vocab)
# 	word_prob = nextword_prob("<s>", lm)

# 	# Laplace Smoothening (by nltk class)
# 	# lpc = Laplace(2)
# 	# lpc.fit(train_tk_pad_bi, vocab)

# 	# built (by class built above)

# 	mlm = My_language_model(train_tk_pad, train_tk_pad_bi, vocab) # flat_pad, n, support
# 	mlm.nxt_words("<s>")
# 	# mlt.perplexity(test)

# 	# perplexity

# 	print("End task1")
