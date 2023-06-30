"""Semantic Vectors
* :function:`.rmv_punctuation`
* :function:`.rmv_stopwords`
* :function:`.lemmatization`
* :function:`.ppmi_weight_matrix`
* :function:`.tfidf_weight_matrix`
* :function:`.semantic_dist`
"""
from nltk.corpus.reader import PlaintextCorpusReader
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import math
# part3
import gensim, logging
from gensim.models import Word2Vec
import gensim.downloader

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



def co_occurence_matrix(words):
	unique = set(words)
	InnerDict = {word:0 for word in unique}
	OutterDict, Dfj, N = {}, InnerDict.copy(), 0
	for i in range(len(words)):
		N += 1 # the total number of context window for all words
		if words[i] in InnerDict.keys():
			OutterDict[words[i]] = {} if words[i] not in OutterDict.keys() else OutterDict[words[i]]
			cache = []
			for j in [-2, -1, 1, 2]: # c_(i-2), c_(i-1), wi, c_(i-+1), c_(i+2)

					if 0 <= i + j < len(words) and words[i] != words[i+j]: # Cc_Matriz: cj not considered context of itself
						try:
							OutterDict[words[i]][words[i+j]] += 1
						except:
							OutterDict[words[i]][words[i+j]] = 1
					if 0 <= i + j < len(words) and words[i] != words[i+j] and words[i+j] not in cache: # consider cj only once
						Dfj[words[i+j]] += 1
						cache.append(words[i+j])
	return OutterDict, Dfj, N

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

def tfidf_weight_matrix(words):
	weights, Dfj, N = co_occurence_matrix(words)
	for w in weights.keys():
		for c in weights[w].keys():
			tf = math.log(weights[w][c]+1)	# Term Frequency smoothing (TF_ij)
			idf = math.log(N/Dfj[c])		# Inverse Document Frequency (IDF_j)
			weights[w][c] = tf * idf
	return weights

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

	corpus_words = rmv_punctuation(raw_words)
	corpus_sents = [rmv_punctuation(sent) for sent in raw_sents]

	# corpus_words = ['mary', 'had', 'a', 'little', 'lamb', 'little', 'lamb', 'little', 'lamb', 'mary', 'had', 'a', 'little', 'lamb', 'whose', 'fleece', 'was', 'white', 'as', 'snow', 'and', 'everywhere', 'that', 'mary', 'went', 'mary', 'went', 'mary', 'went', 'everywhere', 'that', 'mary', 'went', 'the', 'lamb', 'was', 'sure', 'to', 'go']

	# Task 2.1 - TF_IDF
	tf_idf = tfidf_weight_matrix(corpus_words)
	# Task 2.2 - PPMI
	ppmi = ppmi_weight_matrix(corpus_words, 1e-4)
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
		dist = semantic_dist(5, word, tf_idf)
		print("The TF-IDF 5 closest words for '{}', are: {}".format(word, [t[0] for t in dist ]))
