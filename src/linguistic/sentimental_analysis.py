"""Iterpolations
* :function:`.tokenize`
* :function:`.bayesian`
* :function:`.MLE`
* :function:`.Laplace_smoothing`
* :function:`.Perplexity`
* :function:`.aaa`
"""
from collections import Counter

from nltk.corpus.reader import BracketParseCorpusReader

from nltk.probability import FreqDist
from nltk.util import bigrams, ngrams

from nltk.lm import Vocabulary, MLE, models
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

	# Sentences tokenized
	sents_tokenized =[]
	for sent in sents:
		sent_tokenized = [vocab.lookup(word) for word in sent]
		sents_tokenized.append(sent_tokenized)

	sents_flatpad = list(flatten(pad_both_ends(sent, n=2) for sent in sents_tokenized))
	sents_bipad = []
	for i in range(len(sents_tokenized)):
	 	sents_bipad.append(list(bigrams(pad_both_ends(sents_tokenized[i], n=2))))

	if verbose:
		print("The corpus total # of words is {}, of which {} are uniques.".format(len(words),len(set(words))))
		print("The corpus percentage of words OOV is {}.".format(OOV))
		print("The top 5 common works: {}".format(tokens_cnt[:5]))

	return corpus_tokenized, tokens_freq, sents_flatpad, sents_bipad


class laplace:
	""" Creates a laplace smoothed model based on the tokens (not padded) frequency
	the both tokenized and padded ngrams list and tokes list"""
	def __init__(self, tk_freq, ngrams, words) -> None:
		self.word_freq = tk_freq
		self.ngrams_pad = ngrams
		self.words_pad = words

	def __call__(self, h):

		prob = {}
		ngrams_freq = FreqDist(ngrams_pad)

		for bipad in bipad_freq.keys():
			h, w = bipad[0], bipad[1]
			prob[bipad] = bipad_freq[bipad] + 1 / (train_freq[h]+ len(vocab))

		return



def nextword_prob(h, lm, laplace= True,  verbose=True):
	total = 0
	nextword_prob = {}

	for word in vocab:
		if word != h and lm.score(word, [h]) > 0:
			nextword_prob[word] = lm.score(word, [h])
			total += lm.score(word, [h])

	nextword_prob = sorted(nextword_prob.items(), key=lambda x: x[1], reverse=True)
	if verbose:
		print("The top3 most likely words after {}, are:{}".format(h, nextword_prob[:3]))
	return  nextword_prob



	return  nextword_prob


if __name__ == "__main__":

	### TASK 1 ###

	train_corpus = BracketParseCorpusReader(root="corpora", fileids=["train.txt"])
	test_corpus = BracketParseCorpusReader(root="corpora", fileids=["test.txt"])
	support = 3

	# Tokenizing corpus
	train_tk, train_tk_freq, train_tk_pad, train_tk_pad_bi = tokenize(train_corpus, support, True)

	# New vocab considering pad
	vocab = Vocabulary(train_tk_flatpad, unk_cutoff=support)

	# Maximum likelyhood
	lm = MLE(2)
	lm.fit(train_tk_bipad, vocab)

	# Most likely after
	word_prob = nextword_prob("<s>", lm)

	# Laplace smoothening
    lpc = laplace(train_tk_bipad, train_tk_flatpad)


	#word_prob = nextword_prob("<s>", laplace)



	print("End task1")


