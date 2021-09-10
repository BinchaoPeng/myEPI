import copy
import csv
import math
import os
import time
from inspect import signature
from itertools import product

import numpy as np
import pandas as pd
from deepforest import CascadeForestClassifier
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    confusion_matrix, plot_confusion_matrix, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def time_since(start):
    s = time.time() - start
    # s = 62 - start
    if s < 60:
        return '%.2fs' % s
    elif 60 < s and s < 3600:
        s = s / 60
        return '%.2fmin' % s
    else:
        m = math.floor(s / 60)
        s -= m * 60
        h = math.floor(m / 60)
        m -= h * 60
        return '%dh %dm %ds' % (h, m, s)


def performance_result(clf, test_X, test_y):
    best_model = clf
    if hasattr(clf, 'best_estimator_'):
        best_model = clf.best_estimator_

    y_pred = best_model.predict(test_X)
    y_pred_prob_temp = best_model.predict_proba(test_X)
    y_pred_prob = []
    if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
        y_pred_prob = y_pred_prob_temp[:, 0]
    else:
        y_pred_prob = y_pred_prob_temp[:, 1]

    auc = roc_auc_score(test_y, y_pred_prob)
    aupr = average_precision_score(test_y, y_pred_prob)
    acc = accuracy_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    TN, FP, FN, TP = confusion_matrix(test_y, y_pred).ravel()
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    # print("total:", len(test_y), "\tacc num:", (TN + TP), "\tTP:", TP, "\tTN:", TN, "\tFP:", FP, "\tFN:", FN)
    # print("AUC:", auc, end=" ")
    # print("AUPR:", aupr, end=" ")
    # print("acc:", acc, end=" ")
    # print("f1:", f1)
    score_result_dict = {"total": len(test_y), "TP": TP, "TN": TN, "FP": FP, "FN": FN, "precision": precision,
                         "recall": recall,
                         "acc": acc, "AUC": auc, "AUPR": aupr, "F1": f1}
    process_msg = ""
    for k, v in score_result_dict.items():
        if isinstance(v, (int, np.int64)):
            process_msg += "%s: (test=%d) " % (k, v)
        else:
            process_msg += "%s: (test=%.3f) " % (k, v)

    return [score_result_dict, process_msg]


def writeRank2csv(met_grid, clf, index=None):
    print("write rank test to csv!!!")
    csv_rows_list = []
    header = []
    for m in met_grid:
        rank_test_score = 'rank_test_' + m
        mean_test_score = 'mean_test_' + m
        std_test_score = 'std_test_' + m
        header.append(rank_test_score)
        header.append(mean_test_score)
        header.append(std_test_score)
        csv_rows_list.append(clf.cv_results_[rank_test_score])
        csv_rows_list.append(clf.cv_results_[mean_test_score])
        csv_rows_list.append(clf.cv_results_[std_test_score])
    csv_rows_list.append(clf.cv_results_['params'])
    header.append('params')
    results = list(zip(*csv_rows_list))
    print("write over!!!")

    ex_dir_name = '%s_%s_5flod_grid' % (feature_name, method_name)
    if not os.path.exists(r'../../ex/%s/' % ex_dir_name):
        os.mkdir(r'../../ex/%s/' % ex_dir_name)
        os.mkdir(r'../../ex/%s/rank' % ex_dir_name)
        print("created ex folder!!!")
    file_name = r'../../ex/%s/rank/%s_%s_%s_rank_%s.csv' % (ex_dir_name, cell_name, feature_name, method_name, index)
    if index is None:
        file_name = r'../../ex/%s/rank/%s_%s_%s_rank.csv' % (ex_dir_name, cell_name, feature_name, method_name)

    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(results)
        f.close()


def set_estimator_params(estimator, params: dict):
    """
    set estimator parameters
    :param estimator:
    :param params:
    :return:
    """
    # print("set params:", params)
    for k, v in params.items():
        setattr(estimator, k, v)
    return estimator


"""
cell and feature choose
"""
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[2]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest']
method_name = method_names[2]
print("experiment: %s %s_%s" % (cell_name, feature_name, method_name))

trainPath = r'../../data/epivan/%s/features/%s/%s_train.npz' % (cell_name, feature_name, cell_name)
train_data = np.load(trainPath)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
X_en = train_data[train_data.files[0]]
X_pr = train_data[train_data.files[1]]
train_X = [np.hstack((item1, item2)) for item1, item2 in zip(X_en, X_pr)]
# print(type(self.X))
train_y = train_data[train_data.files[2]]
print("trainSet len:", len(train_y))

testPath = r'../../data/epivan/%s/features/%s/%s_test.npz' % (cell_name, feature_name, cell_name)
test_data = np.load(testPath)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
X_en = test_data[test_data.files[0]]
X_pr = test_data[test_data.files[1]]
test_X = [np.hstack((item1, item2)) for item1, item2 in zip(X_en, X_pr)]
test_X = np.array(test_X)
# print(type(self.X))
test_y = test_data[test_data.files[2]]
print("testSet len:", len(test_y))

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

"""
params
"""
parameters = [
    {
        'n_estimators': [2, 4, 6, 8],
        'n_trees': [50, 100, 150, 200],
        'predictors': ['xgboost', 'lightgbm', 'forest'],
        'max_layers': [20, 40, 60, 80],
        'use_predictor': [True]
    },
    {
        'n_estimators': [2, 4, 6, 8],
        'n_trees': [50, 100, 150, 200],
        'max_layers': [20, 40, 60, 80]
    },
]
# parameters = [
#     {
#         'n_estimators': [2, 5],
#         'max_layers': [layer for layer in range(20, 40, 10)],
#         'predictors': ['xgboost', 'lightgbm', 'forest'],
#         'use_predictor': [True]
#     },
# ]
# parameters = [
#
#     {
#         'n_estimators': [2],
#         'max_layers': [30],
#         'predictors': ['xgboost'],
#         'use_predictor': [True]
#     },
#
# ]
# model = CascadeForestClassifier(use_predictor=True, random_state=1, n_jobs=5, predictor='forest')
base_deep_forest = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest', verbose=0)
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
candidate_params = list(ParameterGrid(parameters))
n_candidates = len(candidate_params)
print("Fitting , totalling {0} fits".format(n_candidates))
parallel = Parallel(n_jobs=2)


def fit_and_predict(cand_idx, params):
    process_msg = "[fit {}/{}] END ".format(cand_idx, n_candidates)
    start = time.time()
    deep_forest = copy.deepcopy(base_deep_forest)
    model = set_estimator_params(deep_forest, params)
    model.fit(train_X, train_y)
    params_msg = ""
    for k, v in params.items():
        params_msg += "{}={}, ".format(k, v)

    process_msg += params_msg[: -2] + '; '

    [out, score_result_msg] = performance_result(model, test_X, test_y)
    process_msg += score_result_msg
    process_msg += time_since(start)
    print(process_msg)
    return [out, score_result_msg]


with parallel:
    all_out = []
    out = parallel(delayed(fit_and_predict)
                   (cand_idx, params)
                   for cand_idx, params in enumerate(candidate_params, 1))
    # print(out)
    if len(out) < 1:
        raise ValueError('No fits were performed. '
                         'Was the CV iterator empty? '
                         'Were there no candidates?')
    elif len(out) != n_candidates:
        raise ValueError('cv.split and cv.get_n_splits returned '
                         'inconsistent results. Expected {} '
                         'splits, got {}'
                         .format(n_candidates, len(out)))
    print(out)
    all_out.extend(out)

"""
param doc
{'n_bins': 255, 'bin_subsample': 200000, 'bin_type': 'percentile', 'max_layers': 20, 
'criterion': 'gini', 'n_estimators': 2, 'n_trees': 100, 'max_depth': None, 
'min_samples_leaf': 1, 'predictor_kwargs': {}, 'backend': 'custom', 'n_tolerant_rounds': 2, 
'delta': 1e-05, 'partial_mode': False, 'n_jobs': 5, 'random_state': 1, 'verbose': 0, 'n_layers_': 0, 'is_fitted_': False, 'layers_': {}, 'binners_': {}, 
'buffer_': <deepforest._io.Buffer object at 0x7f84cb1adc50>, 'use_predictor': False, 
'predictor': 'forest', 'labels_are_encoded': False, 'type_of_target_': None, 'label_encoder_': None}

一、更好的准确性
决定是否添加预测器的一个有用规则是将深度森林的性能与从训练数据生成的独立预测器的性能进行比较。
如果预测器始终优于深度森林，那么通过添加预测器，深度森林的性能有望得到改善。
在这种情况下，从深森林产生的增强特征也有助于训练预测器。
1. 增加模型复杂性
n_estimators：指定每个级联层中的估计器数量。
n_trees：指定每个估计器中的树数。
max_layers：指定最大级联层数。
使用上述较大的参数值，深度森林的性能可能会提高复杂数据集的性能，这些数据集需要更大的模型才能表现良好。

2. 添加预测器
use_predictor：决定是否使用连接到深森林的预测器。
predictor: 指定预测器的类型，应为"forest", "xgboost", "lightgbm" 之一


二、更快的速度
由于深度森林根据训练数据的验证性能自动确定模型复杂度，因此将参数设置为较小的值可能会导致具有更多级联层的深度森林模型。
1. 并行化
强烈建议使用并行化，因为深森林自然适合它。
n_jobs：指定使用的工人数量。将其值设置为大于 1 的整数将启用并行化。将其值设置为-1意味着使用所有处理器。

2. 更少的分裂
n_bins：指定特征离散 bin 的数量。较小的值意味着将考虑较少的分裂截止值，应为 [2, 255] 范围内的整数。
bin_type：指定分箱类型。将其值设置为"interval"可以在特征值累积的密集间隔上考虑较少的分割截止点。

3. 降低模型复杂度
将以下参数设置为较小的值会降低深度森林的模型复杂度，并且可能会导致更快的训练和评估速度。
max_depth: 指定树的最大深度。None表示没有约束。
min_samples_leaf：指定叶节点所需的最小样本数。最小值为1。
n_estimators：指定每个级联层中的估计器数量。
n_trees：指定每个估计器中的树数。
n_tolerant_rounds: 指定处理早停时的容忍轮数。最小值为1。

三、降低内存使用率
1. 部分模式
partial_mode：决定是否在部分模式下训练和评估模型。如果设置为True，模型将主动将拟合的估计量转储到本地缓冲区中。
因此，深度森林的内存使用不再随着拟合级联层数的增加而线性增加。

此外，降低模型复杂度也会降低内存使用量。

"""
