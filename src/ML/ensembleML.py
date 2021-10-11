import os
import sys
import time
import warnings
from itertools import product
import numpy as np
from sklearnex import patch_sklearn
import math

patch_sklearn()

warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from xgboost import XGBClassifier
from thundersvm import SVC
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from lightgbm.sklearn import LGBMClassifier

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since, get_scoring_result
from ML.EPIconst import EPIconst


def get_all_data(cell_name, all_feature_names):
    data_ensemble = {}
    for feature_name in all_feature_names:
        data_value = get_data_np_dict(cell_name, feature_name, EPIconst.MethodName.ensemble)
        data_ensemble.update({feature_name: data_value})
    return data_ensemble


def get_new_feature(cell_name, all_feature_names, all_method_names):
    test_y_predict = []
    test_y_proba = []
    train_y_predict = []
    train_y_proba = []
    data_value = {}
    if isinstance(all_method_names, str):
        all_method_names = [all_method_names]
    if isinstance(all_feature_names, str):
        all_feature_names = [all_feature_names]
    for item in product(all_feature_names, all_method_names):
        start_time = time.time()
        ex_item = cell_name + "_" + "_".join(item)
        feature_name = item[0]
        method_name = item[1]
        if ex_item.__contains__("HeLa-S3"):
            ex_item = "HeLa_S3" + "_" + feature_name + "_" + method_name
        model_params = getattr(EPIconst.ModelParams, ex_item)
        base_params = getattr(EPIconst.ModelBaseParams, method_name)
        estimator = estimators[method_name]()
        estimator.set_params(**base_params)
        estimator.set_params(**model_params)
        print(ex_item, ":", estimator)
        data_value = get_data_np_dict(cell_name, feature_name, EPIconst.MethodName.ensemble)
        estimator.fit(data_value["train_X"], data_value["train_y"])
        # get new testSet
        y_pred = estimator.predict(data_value["test_X"])
        y_pred_prob_temp = estimator.predict_proba(data_value["test_X"])
        if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
            y_proba = y_pred_prob_temp[:, 0]
        else:
            y_proba = y_pred_prob_temp[:, 1]
        # print(y_proba)
        test_y_predict.append(y_pred)
        test_y_proba.append(y_proba)
        scoring = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])
        process_msg, score_result_dict = get_scoring_result(scoring, data_value["test_y"], y_pred, y_proba)
        # print(ex_item, ":", process_msg, "\n")
        print("{0}:{1} {2}\n".format(ex_item, process_msg, time_since(start_time)))
        # get new trainSet
        y_pred = estimator.predict(data_value["train_X"])
        y_pred_prob_temp = estimator.predict_proba(data_value["train_X"])
        if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
            y_proba = y_pred_prob_temp[:, 0]
        else:
            y_proba = y_pred_prob_temp[:, 1]
        # print(y_proba)
        train_y_predict.append(y_pred)
        train_y_proba.append(y_proba)

    test_y_pred_np = np.array(test_y_predict)
    test_y_prob_np = np.array(test_y_proba)
    train_y_pred_np = np.array(train_y_predict)
    train_y_prob_np = np.array(train_y_proba)

    train_y = data_value["train_y"]
    print(train_y)
    test_X_pred = test_y_pred_np.T
    test_X_prob = test_y_prob_np.T

    test_y = data_value["test_y"]
    train_X_pred = train_y_pred_np.T
    train_X_prob = train_y_prob_np.T

    # print(test_y_pred_np[0:5, 0:5])
    # print(test_X_pred[0:5, 0:5])
    # print("\n")
    # print(test_y_prob_np[0:5, 0:5])
    # print(test_X_prob[0:5, 0:5])
    # print("\n")
    # print(train_y_pred_np[0:5, 0:5])
    # print(train_X_pred[0:5, 0:5])
    # print("\n")
    # print(train_y_prob_np[0:5, 0:5])
    # print(train_X_prob[0:5, 0:5])
    # print("\n")

    return {'train_X': {"train_X_pred": train_X_pred, "train_X_prob": train_X_prob}, 'train_y': train_y,
            'test_X': {"test_X_pred": test_X_pred, "test_X_prob": test_X_prob}, 'test_y': test_y}


def meta_grid(cell_name, new_feature, datatype, meta_estimator, cv_params, method_name, dir_name, index=None):
    data_list_dict = {}
    if datatype == "pred":
        print("use pred feature !!!")
        data_list_dict = {'train_X': new_feature['train_X']["train_X_pred"], 'train_y': new_feature['train_y'],
                          'test_X': new_feature['test_X']["test_X_pred"], 'test_y': new_feature['test_y']}
    elif datatype == "prob":
        print("use prob feature !!!")
        data_list_dict = {'train_X': new_feature['train_X']["train_X_prob"], 'train_y': new_feature['train_y'],
                          'test_X': new_feature['test_X']["test_X_prob"], 'test_y': new_feature['test_y']}
    elif datatype == "prob_pred":
        print("use prob_pred feature !!!")
        train_X_prob = new_feature["train_X"]["train_X_prob"]
        test_X_prob = new_feature["test_X"]["test_X_prob"]
        train_X_pred = new_feature["train_X"]["train_X_pred"]
        test_X_pred = new_feature["test_X"]["test_X_pred"]
        train_X = np.hstack((train_X_prob, train_X_pred))
        test_X = np.hstack((test_X_prob, test_X_pred))
        train_y = new_feature["train_y"]
        test_y = new_feature["test_y"]
        data_list_dict = {'train_X': train_X, 'train_y': train_y,
                          'test_X': test_X, 'test_y': test_y}
    refit = "roc_auc"
    met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
    clf = RunAndScore(data_list_dict, meta_estimator, cv_params, met_grid, refit=refit, n_jobs=1, verbose=0)

    print("clf.best_estimator_params:", clf.best_estimator_params_)
    print("best params found in line [{1}] for metric [{0}] in rank file".format(refit,
                                                                                 clf.best_estimator_params_idx_ + 2))
    print("best params found in fit [{1}] for metric [{0}] in run_and_score file".format(refit,
                                                                                         clf.best_estimator_params_idx_ + 1))
    print("clf.best_scoring_result:", clf.best_scoring_result)

    writeRank2csv(met_grid, clf, cell_name, datatype, method_name, dir_name, index=index, is_ensemble=True)

    return clf.best_estimator_params_


def ensemble_final(method_name, cv_params, best_params_result=None, index=None):
    meta_estimator = estimators[method_name]()
    base_params = getattr(EPIconst.ModelBaseParams, method_name)
    meta_estimator.set_params(**base_params)
    if best_params_result is not None:
        meta_estimator.set_params(**best_params_result)
    print("ensemble model params:", meta_estimator.get_params())
    best_estimator_params = meta_grid(cell_name, new_feature, "prob_pred", meta_estimator, cv_params=cv_params,
                                      method_name=method_name,
                                      dir_name=dir_name, index=index)
    return best_estimator_params


if __name__ == '__main__':
    """
    cell and feature choose
    """
    estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier,
                  "deepforest": CascadeForestClassifier,
                  "lightgbm": LGBMClassifier}
    names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
    cell_name = names[3]
    feature_names = ['pseknc', 'cksnap', 'dpcp', 'eiip', 'kmer', 'dnabert_6mer', 'longformer-hug', 'elmo']
    feature_name = feature_names[4]
    method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm', 'rf']
    method_name = method_names[0]
    dir_name = "ensemble"
    ex_dir_name = '%s_%s_%s' % (feature_name, method_name, dir_name)
    if not os.path.exists(r'../../ex_stacking/%s/' % ex_dir_name):
        os.mkdir(r'../../ex_stacking/%s/' % ex_dir_name)
        print("created ex folder!!!")
    if not os.path.exists(r'../../ex_stacking/%s/rank' % ex_dir_name):
        os.mkdir(r'../../ex_stacking/%s/rank' % ex_dir_name)
        print("created rank folder!!!")
    s_time = time.time()
    new_feature = get_new_feature(cell_name, EPIconst.FeatureName.all,
                                  EPIconst.MethodName.all)
    # train_X_prob = new_feature["train_X"]["train_X_prob"]
    # test_X_prob = new_feature["test_X"]["test_X_prob"]
    # train_X_pred = new_feature["train_X"]["train_X_pred"]
    # test_X_pred = new_feature["test_X"]["test_X_pred"]
    # train_X = np.hstack((train_X_prob, train_X_pred))
    # test_X = np.hstack((test_X_prob, test_X_pred))
    # train_y = new_feature["train_y"]
    # test_y = new_feature["test_y"]
    print("\nfeature time: ", time_since(s_time), "\n")
    ##############################################################
    # deepforest
    #############################################################
    method_name = EPIconst.MethodName.deepforest
    deepforest_parameters = [
        {
            'n_estimators': [2, 5, 8, 10, 13],
            'n_trees': [50, 100, 150, 200, 250, 300, 400],
            # 'max_layers': [10, 15, 20, 50, 80, 120],
            'max_layers': [10, 15, 20, 25],
        },
    ]
    ensemble_final(method_name, deepforest_parameters)
    ##############################################################
    # lightgbm
    #############################################################
    start_time = time.time()
    method_name = EPIconst.MethodName.lightgbm
    best_params_result = {}
    # 第一次：max_depth、num_leaves
    print("第一次")
    cv_params = {'max_depth': [-1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'num_leaves': range(221, 350, 10)}
    # cv_params = {'max_depth': [-1], 'num_leaves': [191]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '1')
    best_params_result.update(best_params)
    # 第二次
    print("第二次")
    cv_params = {'max_bin': range(5, 256, 10), 'min_child_samples': range(10, 201, 10)}
    # cv_params = {'max_bin': range(5, 256, 100), 'min_child_samples': range(1, 102, 50)}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '2')
    best_params_result.update(best_params)
    # print( )
    best_params_result.update(best_params)

    # 第三次
    print("第三次")
    cv_params = {'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                 'subsample_freq': range(0, 81, 10)
                 }
    # cv_params = {'colsample_bytree': [0.6, 0.7, ],
    #              'subsample': [0.6, 0.7, ],
    #              'subsample_freq': range(0, 81, 40)
    #              }
    best_params = ensemble_final(method_name, cv_params, best_params_result, '3')
    best_params_result.update(best_params)
    # print( )
    best_params_result.update(best_params)

    # 第四次
    print("第四次")
    cv_params = {'reg_alpha': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                 'reg_lambda': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                 }
    # cv_params = {'reg_alpha': [1e-5, 1e-3, ],
    #              'reg_lambda': [1e-5, 1e-3, ]
    #              }
    best_params = ensemble_final(method_name, cv_params, best_params_result, '4')
    best_params_result.update(best_params)
    # print( )
    best_params_result.update(best_params)

    # 第五次
    print("第五次")
    cv_params = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    # cv_params = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '5')
    best_params_result.update(best_params)
    # print( )
    best_params_result.update(best_params)

    # 第六次
    print("第六次")
    cv_params = {'learning_rate': [0.001, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.75, 1.0],
                 'n_estimators': range(50, 251, 25)}
    # cv_params = {'learning_rate': [0.01, 0.05, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '6')
    best_params_result.update(best_params)
    # print( )
    best_params_result.update(best_params)

    print("total time spending:", time_since(start_time))
    print("best_params_result:", best_params_result)
    ##############################################################
    # rf
    #############################################################
    start_time = time.time()
    method_name = EPIconst.MethodName.rf
    best_params_result = {}
    # 第一次：决策树的最佳数量也就是估计器的数目
    print("第一次")
    cv_params = {'n_estimators': list(range(10, 350, 10))}
    # cv_params = {'n_estimators': [120]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '1')
    best_params_result.update(best_params)
    # 第二次
    print("第二次")
    max_depth = [None]
    max_depth.extend((list(range(1, 150))))
    cv_params = {'max_depth': max_depth}
    # cv_params = {'max_depth': [99]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '2')
    # print( best_params_result)
    best_params_result.update(best_params)

    # 第三次
    print("第三次")
    cv_params = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10], "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    # cv_params = {'gamma': [0.1, 0.2, 0.3,]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '3')
    # print( best_params_result)
    best_params_result.update(best_params)

    # 第四次
    print("第四次")
    cv_params = {'max_features': ["auto", "sqrt", "log2", None]}
    # cv_params = {'max_features': [0.6, 0.7, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '4')
    # print( best_params_result)
    best_params_result.update(best_params)

    print("total time spending:", time_since(start_time))
    print("best_params_result:", best_params_result)

    ##############################################################
    # svm
    #############################################################
    method_name = EPIconst.MethodName.svm
    svm_cv_params = [
        {
            # 'C': [math.pow(2, i) for i in range(-10, 15)],
            'C': [math.pow(2, i) for i in range(-4, 11)],
            # 'gamma': [math.pow(2, i) for i in range(-10, 15)],
            'gamma': [math.pow(2, i) for i in range(-4, 12)],
            'kernel': ['rbf']
        },
    ]
    ensemble_final(method_name, svm_cv_params)

    ##############################################################
    # xgboost
    #############################################################
    start_time = time.time()
    best_params_result = {}
    # 第一次：决策树的最佳数量也就是估计器的数目
    print("第一次")
    cv_params = {'n_estimators': list(range(50, 1050, 50))}
    # cv_params = {'n_estimators': list(range(50, 300, 50))}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '1')
    best_params_result.update(best_params)

    # 第二次
    print("第二次")
    cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # cv_params = {'max_depth': [3, 4, ], 'min_child_weight': [1, 2, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '2')
    # print(best_params_result)
    best_params_result.update(best_params)

    # 第三次
    print("第三次")
    cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # cv_params = {'gamma': [0.1, 0.2, 0.3,]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '3')
    # print(best_params_result)
    best_params_result.update(best_params)

    # 第四次
    print("第四次")
    cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # cv_params = {'subsample': [0.6, 0.7, ], 'colsample_bytree': [0.6, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '4')
    # print(best_params_result)
    best_params_result.update(best_params)

    # 第五次
    print("第五次")
    cv_params = {'reg_alpha': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3],
                 'reg_lambda': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3]}
    # cv_params = {'reg_alpha': [0.05, ], 'reg_lambda': [0.05, 0.1, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '5')
    # print(best_params_result)
    best_params_result.update(best_params)

    # 第六次
    print("第六次")
    cv_params = {'learning_rate': [0.001, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.75, 1.0]}
    # cv_params = {'learning_rate': [0.01, 0.05, ]}
    best_params = ensemble_final(method_name, cv_params, best_params_result, '6')
    # print(best_params_result)
    best_params_result.update(best_params)

    print("total time spending:", time_since(start_time))
    print("best_params_result:", best_params_result)
