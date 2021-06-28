"""Optimization
* :function:`.single_model_nested_cv`
* :function:`.double_model_nested_cv`
"""

def nested_single_cv(x_t, y_t, L, grid, k_ext, k_int):
    """
    Summary:    Help set a hyper-parameters list for a given model before makes
    --------    its comparison with others hyper-parameterized models.

    Input:      - x_t: features train (numpy.arrays)
    ------      - y_t: labels train (numpy.arrays)
                - L: learning algorithm (class method .predict())
                - grid: keys as a parameter name; values as the array of the parameter' values (dict)
                - K_ext: number of external folds (integer)
                - K_int: number of internal folds (integer)

    Output:      - inner_result_frame: index: [k_ext], columns: [hp_set], values: [v_bcr_(k_int_mean)]
    -------      - outter_result_frame: index: [k_ext, hp_hat], columns:[t_bcr, v_bcr], values:[t_bcr, v_bcr]

    Example:    model1= BaggingTrees
    --------    grid1 = {'epochs':[1]
                                    , 'n_trees':[100]
                                    , 'criterion': ['entropy']
                                    , 'min_samples_leaf':[0.06] #
                                    , 'max_depth':[3]
                                    , 'min_samples_split':[0.03] #
                                    , 'max_leaf_nodes':[200]
                                    }

                K_int, K_ext = 4, 10
                outter, inner = nested_single_cv(x_t, y_t, model1, grid1, K_ext, K_int)

                outter.groupby('hp_hat').agg({'t_bcr': ['count', 'mean', 'std']
                                            , 'v_bcr': ['mean', 'std']}).reset_index('hp_hat')


    """

    hp_set = [v for v in product(*grid.values())]

    inner_results = pd.DataFrame(columns = hp_set)

    outter_results = pd.DataFrame(columns = ['hp_hat'
                                        , 't_bcr'
                                        , 'v_bcr'
                                        ])


    # frame pointer
    i = 0
    # partionate "training rows" into "K_ext" sets
    K_ext_folds = KFold(n_splits = k_ext, shuffle=False).split(x_t)  # (markers t_i, v_i)
    for t_ext_fold, v_ext_fold in K_ext_folds:
        # sectioning "train set" between "S_k" into "ext_fold" sets
        x_S_k = x_t[t_ext_fold] # training x
        y_S_k = y_t[t_ext_fold] # training y
        x_ext_fold = x_t[v_ext_fold] # test x
        y_ext_fold = y_t[v_ext_fold] # test y

        # get hp_hat in the inner loop
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
        # # Averages the k_int iteractions for each hp in hp_set and stores it
        inner_results.loc[i] = [sum(arr) / len(arr) for arr in hp_dic.values()]


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

        # training metrics
        t_acc = model.acc
        t_bcr = model.bcr
        t_f1 = model.f1
        t_auc = model.auc
        # validation metrics
        v_acc = accuracy_score(Y_v, Y_v_predicted)
        v_bcr = balanced_accuracy_score(Y_v, Y_v_predicted)
        v_f1 = f1_score(Y_v, Y_v_predicted, average='macro')
        v_auc = roc_auc_score(Y_v, Y_v_predicted, average='macro')

        outter_results.loc[i] = [hp_hat
                                , t_bcr
                                , v_bcr]
        i += 1


    return outter_results, inner_results
