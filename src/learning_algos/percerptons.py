
"""Perceptrons
* :class:`.perceptron_margin`
* :function:`.predict`
"""

import pandas as pd
import numpy as np


def perceptron_margin(X, y, b, w_init, eta, epoch):
	'''
	-->> Transform into class
	updated performed by epoch (#lines traininng set)
	stops after epoch iteraction or convergence
	b: margin

	'''
	# adding column of 1s for the w_0 (bias or intersections)
	X = np.concatenate([np.ones((X.shape[0], 1)),X], axis=1)
	rows = X.shape[0]
	cols = X.shape[1]
	# initializing w with size cols as column vector
	if not w_init:
		w_init = np.array([0 for i in range(cols)])
	w =  w_init
	# pointing all summation towards 1
	for i in range(X.shape[0]):
		X[i] = y[i]*X[i]

	k = 0
	while k < epoch:
		i = k % rows
		xi = X[i]
		# row_vector*column_vector
		summation = np.dot(xi, w)
		if summation <= b :
			w = np.add(w,np.multiply(eta,xi))
		# matrix [row,col] * column vector
		predictions = np.dot(X, w)
		# column vector interaction
		if (predictions > b).all():
			return w
		k += 1
	return w



def predict(w, X):

    predictions = []
    for xi in X:
        net_input = np.dot(xi, w[1:]) + w[0]
        prediction = 1 if (net_input >= 0) else -1
        predictions.append(prediction)
    return predictions

# ### exmaple test

# path= '/mnt/c/Users/b_tib/coding/Msc/oLINGI2262/Inginious_task_2/Linear_Discriminant_Classifiers'

# train_df = pd.read_csv(path + '/RestrictedLettersTrain.csv', index_col=0)
# x_train = np.array(train_df.iloc[:,:-1])
# y_train = np.array(train_df['labels'].replace(['A','E'],[1,-1]))

# valid_df = pd.read_csv(path + '/RestrictedLettersValid.csv', index_col=0)
# x_valid = np.array(valid_df.iloc[:,:-1])
# y_valid = np.array(valid_df['labels'].replace(['A','E'],[1,-1]))

# test_df = pd.read_csv(path + '/RestrictedLettersTest.csv', index_col=0)
# x_test = np.array(test_df.iloc[:,:-1])
# y_test = np.array(test_df['labels'].replace(['A','E'],[1,-1]))



# w1 = perceptron_margin(X=x_train, y=y_train, b=0.1, w_init=None, eta=.2, epoch=10)


# w1_train_pred = predict(w1, x_train)

# print(w1)
# print(w1_train_pred)


