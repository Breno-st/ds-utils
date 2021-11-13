"""Sentimental analysis
* :class:`.My_language_model`
* :function:`.tokenize`
* :function:`.processing_functions`
"""

from collections import Counter
import numpy as np
import math

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.util import bigrams, ngrams
from nltk.lm import Vocabulary, MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten

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
# processing f1
def rmv_punctuation(array):
	"Remove punctuation from a [sentence]"
	return [x.lower() for x in array if x.isalpha()]
# processing f2
def rmv_stopwords(array):
	"Remove 'stop words' from a [sentence]"
	return [x.lower() for x in array if not x in stopwords.words("english")]
# processing f3
def lemmatization(array):
	"Remove use lematization for words in [sentence]"
	lemma = WordNetLemmatizer()
	return [lemma.lemmatize(x, pos = "v") for x in [lemma.lemmatize(word, pos = "n") for word in array]]
# processing f4
def stemming(array):
	"Remove use steeming for words in [sentence]"
	pass
	return


