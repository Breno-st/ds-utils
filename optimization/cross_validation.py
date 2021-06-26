
"""Optimization
* :function:`.single_model_nested_cv`
* :function:`.double_model_nested_cv`
"""

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


def nested_single_cv(x_t, y_t, L, k_ext, k_int, hp_set):
    """
    Help set a hyper-parameters list for a given model before makes
    its comparison with others hyper-parameterized models.

    Input:
        - x_t: features train (numpy.arrays)
        - y_t: labels train (numpy.arrays)
        - K_ext: number of external folds (integer)
        - K_int: number of internal folds (integer)
        - L: learning algorithm function (function or class)
        - hp_set: list of parameters of the learning algorithm (array)

    Output:
        - Dataframe:
            - train_acc mean
            - train_acc std
            - train_f1 mean

    """


    results_frame = pd.DataFrame(columns =['hp_hat'
                                        , 't_bcr'
                                        , 'v_bcr'])

    # frame pointer
    i = 0
    # experiemental 3 metrics each one with mean & std
    outer = np.empty((0,6), float)
    # partionate "training rows" into "K_ext" sets
    K_ext_folds = KFold(n_splits = k_ext, shuffle=False).split(x_t)  # (markers t_i, v_i)
    for t_ext_fold, v_ext_fold in K_ext_folds:
        # sectioning "train set" between "S_k" into "ext_fold" sets
        x_S_k = x_t[t_ext_fold] # training x
        y_S_k = y_t[t_ext_fold] # training y
        x_ext_fold = x_t[v_ext_fold] # test x
        y_ext_fold = y_t[v_ext_fold] # test y


        # get hp_hat in the inner loop
        hp_hat, hp_mean = None, 0
        hp_dic = {}
        for idx, hp in enumerate(hp_set):
            hp_dic[idx]=[]
            # partionate "S_k training rows" into "K_int" sets
            K_int_folds = KFold(n_splits = k_int, shuffle=False).split(x_S_k)
            for t_int_fold, v_int_fold in K_int_folds:
                # sectioning "S_k" between "Ss_k" into "int_fold" sets
                x_Ss_k = x_S_k[t_int_fold] # training x
                y_Ss_k = y_S_k[t_int_fold] # training y
                x_int_fold = x_S_k[v_int_fold] # test x
                y_int_fold = y_S_k[v_int_fold] # test y

                # must scaler after partition, for specific a training normalization
                min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                X_t = min_max_scaler.fit_transform(x_Ss_k)
                X_v = min_max_scaler.fit_transform(x_int_fold)
                Y_t = y_Ss_k
                Y_v = y_int_fold

                # Loading and fitting model
                model = L(hp)
                model.fit(X_t, Y_t)
                # prediction
                Y_v_predicted = model.predict(X_v)
                # validation
                v_bcr = balanced_accuracy_score(Y_v, Y_v_predicted)
                # append all
                hp_dic[idx].append(v_bcr)

        # avg all hp predictions scores and define the higher to hp_hat
        ixd_max= max([(k,np.mean(v)) for k,v in hp_dic.items()],key=lambda item:item[1])[0]
        hp_hat = hp_set[ixd_max]

        # must scaler after partition, for specific a training normalization
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        X_t = min_max_scaler.fit_transform(x_S_k)
        X_v = min_max_scaler.fit_transform(x_ext_fold)
        Y_t = y_S_k
        Y_v = y_ext_fold

        # Loading and fitting model
        model = L(hp)
        model.fit(X_t, Y_t)
        # prediction
        Y_v_predicted = model.predict(X_v)
        Y_t_predicted = model.predict(X_t)

        # validation
        t_bcr = balanced_accuracy_score(Y_t, Y_t_predicted)
        v_bcr = balanced_accuracy_score(Y_v, Y_v_predicted)

        #    # other validation metrics
        #     v_acc = accuracy_score(Y_v, Y_v_predicted)
        #     v_f1 = f1_score(Y_v, Y_v_predicted, average='macro')
        #     v_auc = roc_auc_score(Y_v, Y_v_predicted, average='macro')


        results_frame.loc[i] = [hp_hat
                                , t_bcr
                                , v_bcr]
        i += 1


    return results_frame
