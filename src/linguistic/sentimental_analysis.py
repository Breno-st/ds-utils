"""Iterpolations
* :function:`.tokenize`
* :class:`.laplace`
* :function:`.bayesian`
* :function:`.Perplexity`
"""
from collections import Counter
import numpy as np

from nltk.corpus.reader import BracketParseCorpusReader

from nltk.probability import FreqDist
from nltk.util import bigrams, ngrams

from nltk.lm import Vocabulary, MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten


def tokenize(corpus, support, verbose=False):
	""" Tokenize corpus according to a support and returning
	toknes and a dictionair
	"""
	words, sents = corpus.words(), corpus.sents()

	# Corpus Vocabulary
	vocab = Vocabulary(words, unk_cutoff=support) # trying corpus instaed of words
	corpus_tokenized = vocab.lookup(words) # list of tokens repect to vocab

	# Words percentage defined as '<UNK>'
	OOV = round(100*Counter(corpus_tokenized)['<UNK>']/len(words),3)

	# Words frequency
	tokens_freq = FreqDist(corpus_tokenized)
	tokens_cnt = [(k,v) for k,v in tokens_freq.items() if v >=support]
	tokens_cnt.sort(key = lambda x:x[1], reverse = True )

	# Tokenizin sentences
	sents_tokenized =[]
	for sent in sents:
		sent_tokenized = [vocab.lookup(word) for word in sent]
		sents_tokenized.append(sent_tokenized)

	# Padding tokenized sentences
	sents_flatpad = list(flatten(pad_both_ends(sent, n=2) for sent in sents_tokenized))
	sents_bipad = []
	for i in range(len(sents_tokenized)):
	 	sents_bipad.append(list(bigrams(pad_both_ends(sents_tokenized[i], n=2)))) # necessary bi-gram???

	if verbose:
		print("The corpus total # of words is {}, of which {} are uniques.".format(len(words),len(set(words))))
		print("The corpus percentage of words OOV is {}.".format(OOV))
		print("The 10th tokenized bigram sentence {}.".format(sents_bipad[9]))
	return corpus_tokenized, tokens_freq, sents_flatpad, sents_bipad


class Laplace_:
	""" ONLY Bi-grams
	Creates a laplace smoothed model based on the tokens (not padded) frequency
	the both tokenized and padded ngrams list and tokes list
	input: 	- tokens padded    = MLE loaded with bigram(tk_pad) and vocab(tk_pad)
		 	- tokens padded bi = vocab = lm.vocab
			- vocabulary       = token (by suppor) vocab

	For ngrams recussions: https://github.com/joshualoehr/ngram-language-model/blob/master/language_model.py
	"""
	def __init__(self, tks_pad, tks_bigrams_pad, vocab) -> None:
		self.tokens = tks_pad
		self.bigrams = tks_bigrams_pad
		self.vocab = vocab
	# def _smooth(self):
	# 	tks_pad, tks_bigrams_pad, vocab = self.tokens, self.bigrams, self.vocab

		flat_tks_bigrams_pad = []
		for l in tks_bigrams_pad:
			flat_tks_bigrams_pad += l

		tks_freq = FreqDist(tks_pad)
		bigrams_freq = FreqDist(flat_tks_bigrams_pad)

		smoothed_cnt = {}
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



if __name__ == "__main__":

	### TASK 1 ###

	train_corpus = BracketParseCorpusReader(root="corpora", fileids=["train.txt"])
	test_corpus = BracketParseCorpusReader(root="corpora", fileids=["test.txt"])
	support = 3

	# Tokenizing corpus
	train_tk, train_tk_freq, train_tk_pad, train_tk_pad_bi = tokenize(train_corpus, support, True)


	# Tokenized vocabulary
	vocab = Vocabulary(train_tk_pad, unk_cutoff=support)

	# Maximum likelyhood (by nltk class)
	lm = MLE(2)
	lm.fit(train_tk_pad_bi, vocab)
	# lm.count(['a'])
	# lm.count(['a']['b']) # ab
	# lm.score('a')
	# lm.score('b', ['a']) # ab
	# lm.train = train_tk_pad_bi

	# # Laplace smoothening (by nltk class)
	# lpc = Laplace(2)
	# lpc.fit(train_tk_pad_bi, vocab)

	# built (by class built above)
	lpc_ = Laplace_(train_tk_pad, train_tk_pad_bi, vocab) #

	# Most likely after
	word_prob = nextword_prob("<s>", lm)
	word_prob = nextword_prob("<s>", lpc_)

	# perplexity



	print("End task1")


