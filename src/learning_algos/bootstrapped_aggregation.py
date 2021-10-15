"""Bootstrapped_agregation
* :class:`.BaggingTress`
* :class:`.BaggingSVC`
"""

# data wrangling
import numpy as np
import pandas as pd
from collections import Counter

# models
from sklearn import  svm
from sklearn.tree import DecisionTreeClassifier

# validation
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score


class BaggingTrees:
    """
    Summary:        Expand the subset of features in regards each node split for
    --------        a more flexible tunning.

    Parameters:     - X_t : Training set features. (np.array)
    ----------      - Y_t : Training set labels. (np.array)
                    - X_v : Validation set features. (np.array)
                    - p :   List of parameters values:
                            -- epochs: generates statiscal inferences as mean and std
                            -- n_trees: number of tress bagged
                            -- criterion: decision-tree sklearn library
                            -- min_samples_leaf: decision-tree sklearn library
                            -- max_depth: decision-tree sklearn library
                            -- min_samples_splits: decision-tree sklearn library
                            -- max_leaf_nodes: decision-tree sklearn library

    Output:         - y_pred:      predictions on validation set X_v (array)
    -------         - unan_rates:  rate of majority votes (array)
                    - acc:         accuracy on training set Y_t (integer)
                    - f1:          f1 score on training set Y_t (integer)


    Example:        model = BaggingTrees(p)
    --------        model.fit(x_t, y_t)
                    predictions = model.predict(x_v)

                    votes_percentages = model.votes
                    training_accuracy = model.acc
                    training_bcr = model.bcr
                    training_f1 = model.f1
                    training_auc = model.auc

    """

    def __init__(self, p):

        # store parameters
        self.epochs = p[0]; self.n_trees = p[1]
        self.criterion = p[2]; self.min_samples_leaf = p[3]
        self.max_depth = p[4]; self.min_samples_splits = p[5]
        self.max_leaf_nodes = p[6]


    def fit(self, X_t, Y_t):

        if isinstance(X_t,np.ndarray):
            X_t = pd.DataFrame(X_t)
        elif not isinstance(X_t,pd.core.frame.DataFrame):
            raise Exception('Wrong type for X_t. Expected np.ndarray or pd.DataFrame')

        if isinstance(Y_t,np.ndarray):
            Y_t = pd.DataFrame(Y_t)
        elif not isinstance(X_v,pd.core.frame.DataFrame):
            raise Exception('Wrong type for Y_t. Expected np.ndarray or pd.DataFrame')

        self.X_t_df = X_t.copy(); self.Y_t_df = Y_t.copy()

        X_t['label'] = Y_t
        train_df = X_t
        for i in range(self.epochs):
            self.bag = []
            for run in np.arange(self.n_trees):
                # resampling the dataframe (number of distinct, number of distinct)
                train_df_bs = train_df.iloc[np.random.randint(len(train_df), size=len(train_df))]
                X_train = train_df_bs.iloc[:,:-1]
                Y_train = train_df_bs.iloc[:,-1:]
                # Storing each trained tree
                wl = DecisionTreeClassifier(criterion=self.criterion
                                        , min_samples_leaf=self.min_samples_leaf
                                        , max_depth=self.max_depth
                                        , min_samples_split=self.min_samples_splits
                                        , max_leaf_nodes=self.max_leaf_nodes).fit(X_train,Y_train)
                                        #, random_state=run
                # add tree into bag
                self.bag.append(wl)

            ## Score on Training set
            t_predictions = []
            for i in range(self.n_trees):
                tree_t_prediction = self.bag[i].predict(self.X_t_df) # predict validation and training sets
                t_predictions.append(tree_t_prediction) # Append predictions

            # Convert predictions lists into np.array to transpose them and obtain "n_tree" predictions per line
            t_predictions_T = np.array(t_predictions).T

            t_final_predictions = []
            # for each entry "m" of X_t_df(m x features)
            for line in t_predictions_T:
                # countabilize the "n_tree" votes in v_predictions_T (m x n_tree)
                most_common = Counter(line).most_common(1)[0][0]
                t_final_predictions.append(most_common)

            # accuracies values
            self.acc = accuracy_score(self.Y_t_df, t_final_predictions)
            self.f1 = f1_score(self.Y_t_df, t_final_predictions, average='macro')
            self.bcr = balanced_accuracy_score(self.Y_t_df, t_final_predictions)
            self.auc = roc_auc_score(self.Y_t_df, t_final_predictions, average='macro')

        return

    def predict(self, X_v):

        if isinstance(X_v,np.ndarray):
            X_v = pd.DataFrame(X_v)
        elif not isinstance(X_v,pd.core.frame.DataFrame):
            raise Exception('Wrong type for X_v. Expected np.ndarray or pd.DataFrame')

        self.X_v_df = X_v.copy()
        ## Prediction on Validation set
        v_predictions = []
        # each tree will make a prediction about test_df
        for i in range(self.n_trees):
            tree_v_prediction = self.bag[i].predict(self.X_v_df) # predict validation and training sets
            v_predictions.append(tree_v_prediction) # Append predictions
        # Convert predictions lists into np.array to transpose them and obtain "n_tree" predictions per line
        v_predictions_T = np.array(v_predictions).T

        self.prediction = []
        self.votes = []
        # for each entry "n" of X_v_df(n x features)
        for line in v_predictions_T:
            # countabilize the "n_tree" votes in v_predictions_T (n x n_tree)
            most_common = Counter(line).most_common(1)[0][0]
            unanimity_rate = Counter(line)[most_common] / len(line)
            # get prediction and unanimity rate
            self.prediction.append(most_common)
            self.votes.append(unanimity_rate)
        return self.prediction


class BaggingSVC:
    """
    Summary:        Expand the subset of features in regards each node split for
    --------        a more flexible tunning.

    Parameters:     - X_t : Training set features. (np.array)
    ----------      - Y_t : Training set labels. (np.array)
                    - X_v : Validation set features. (np.array)
                    - p :   List of parameters values:
                            -- epochs: generates statiscal inferences as mean and std
                            -- n_trees: number of tress bagged
                            -- criterion: svm.svc sklearn library
                            -- min_samples_leaf: svm.svc sklearn library
                            -- max_depth: svm.svc sklearn library
                            -- min_samples_splits: svm.svc sklearn library
                            -- max_leaf_nodes: svm.svc sklearn library

    Output:         - y_pred:      predictions on validation set X_v (array)
    -------         - unan_rates:  rate of majority votes (array)
                    - acc:         accuracy on training set Y_t (integer)
                    - f1:          f1 score on training set Y_t (integer)


    Example:        model = BaggingSVC(p)
    --------        model.fit(x_t, y_t)
                    predictions = model.predict(x_v)

                    votes_percentages = model.votes
                    training_accuracy = model.acc
                    training_bcr = model.bcr
                    training_f1 = model.f1
                    training_auc = model.auc
    """
    # TODO
