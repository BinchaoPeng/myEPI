import os
import sys
import time
import warnings
from itertools import product
import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
start_time = time.time()
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

estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier, "deepforest": CascadeForestClassifier,
              "lightgbm": LGBMClassifier}

names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[0]


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
    for item in product(all_feature_names, all_method_names):
        ex_item = cell_name + "_" + "_".join(item)
        feature_name = item[0]
        method_name = item[1]
        print("ensemble_ex_item:", ex_item)
        if ex_item.__contains__("HeLa-S3"):
            ex_item = "HeLa_S3" + "_" + feature_name + "_" + method_name
        model_params = getattr(EPIconst.Params, ex_item)
        base_params = getattr(EPIconst.BaseParams, method_name)
        estimator = estimators[method_name]()
        estimator.set_params(**base_params)
        estimator.set_params(**model_params)
        print(estimator)
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
        print(ex_item, ":", process_msg, "\n")
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



if __name__ == '__main__':
    EPIconst.MethodName.all.remove("rf")
    new_feature = get_new_feature(EPIconst.CellName.HUVEC, EPIconst.FeatureName.all,
                                  EPIconst.MethodName.all)
    train_X_prob = new_feature["train_X"]["tain_X_prob"]
    test_X_prob = new_feature["test_X"]["test_X_prob"]
    train_X_pred = new_feature["train_X"]["tain_X_pred"]
    test_X_pred = new_feature["test_X"]["test_X_pred"]
    train_X = np.hstack((train_X_prob, train_X_pred))
    test_X = np.hstack((test_X_prob, test_X_pred))
    train_y = new_feature["train_y"]
    test_y = new_feature["test_y"]


