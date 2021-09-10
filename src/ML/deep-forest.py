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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
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


class MyGridSearchCV:

    def __init__(self, train_X, train_y, estimator, parameters, scoring, refit, n_jobs, cv):
        self.n_jobs = n_jobs
        self.cv = cv
        self.estimator = estimator
        self.scoring = scoring
        self.refit = refit

        self.data_list = self.get_CV_data(train_X, train_y)
        self.candidate_params = self.get_CV_candidate_params(parameters)
        self.all_out = self.run_search()
        self.cv_results_ = self.get_cv_results()
        self.best_estimator_params_ = self.get_best_estimator_params()
        self.best_estimator_ = self.get_best_estimator()

    """
    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    [CV 3/5] END n_estimators=50; accuracy: (test=0.847) average_precision: (test=0.922) f1: (test=0.843) roc_auc: (test=0.924) total time=   4.6s
    """

    def get_CV_data(self, X, y):
        """
        get train and test data of CV
        :param X:
        :param y:
        :param cv:
        :return:
        """
        kf = StratifiedKFold(n_splits=self.cv, shuffle=False)
        # kf.get_n_splits(X)
        # print(kf)
        data_list = []
        for index, item in enumerate(kf.split(X, y), 1):
            train_index, test_index = item
            # print("TRAIN:", train_index, "TEST:", test_index)
            train_X, test_X = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]
            data = {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}
            data_list.append(data)
        return data_list

    def get_CV_candidate_params(self, parameters):
        """
        get candidate_params
        :param parameters:
        :return:
        """
        return list(ParameterGrid(parameters))

    def model_fit(self, estimator, train_X, train_y):
        """
        train model
        :param estimator:
        :param train_X:
        :param train_y:
        :return:
        """
        estimator.fit(train_X, train_y)

    def model_predict(self, model, test_X):
        """
        get y_pred
        :param model:
        :return:
        """
        y_pred = model.predict(test_X)
        return y_pred

    def model_predict_prob(self, model, test_X):
        """
        get y_prob_temp
        :param model:
        :return:
        """
        y_prob_temp = model.predict_proba(test_X)
        return y_prob_temp

    def set_estimator_params(self, estimator, params: dict):
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

    def _fit_and_score(self, split_idx, params, data):
        start_time = time.time()
        process_msg = "[CV {}/{}] END ".format(split_idx, self.cv)

        estimator = copy.deepcopy(self.estimator)
        estimator = self.set_estimator_params(estimator, params)
        # print("estimator:", estimator.__dict__)
        self.model_fit(estimator, train_X=data['train_X'], train_y=data['train_y'])
        y_pred = self.model_predict(estimator, test_X=data['test_X'])
        y_pred_prob_temp = self.model_predict_prob(estimator, test_X=data['test_X'])

        if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
            y_prob = y_pred_prob_temp[:, 0]
        else:
            y_prob = y_pred_prob_temp[:, 1]
        process_msg += self.get_params_msg(params)
        score_result_msg, score_result_dict = self.get_scoring_result(data['test_y'], y_pred, y_prob)
        process_msg += score_result_msg
        total_time = time_since(start_time)
        process_msg += "total time=  {}".format(total_time)
        print(process_msg)

        return [params, score_result_dict]

    def run_search(self, pre_dispatch='2 * n_jobs', batch_size='auto'):
        """
        run cv search with parallel
        :param pre_dispatch:
        :param batch_size:
        :return:
        """
        n_candidates = len(self.candidate_params)
        n_splits = self.cv
        print("Fitting {0} folds for each of {1} candidates,"
              " totalling {2} fits".format(self.cv, n_candidates, self.cv * n_candidates))
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=pre_dispatch, batch_size=batch_size)

        with parallel:
            all_out = []
            out = parallel(delayed(self._fit_and_score)
                           (split_idx, params, data)
                           for (cand_idx, params),
                               (split_idx, data) in product(enumerate(self.candidate_params, 1),
                                                            enumerate(self.data_list, 1)))
            # print(out)
            if len(out) < 1:
                raise ValueError('No fits were performed. '
                                 'Was the CV iterator empty? '
                                 'Were there no candidates?')
            elif len(out) != n_candidates * n_splits:
                raise ValueError('cv.split and cv.get_n_splits returned '
                                 'inconsistent results. Expected {} '
                                 'splits, got {}'
                                 .format(n_splits,
                                         len(out) // n_candidates))

            all_out.extend(out)
            # print(all_out)
        return all_out

    def _get_score_dict(self):
        score_dict = {}
        if isinstance(self.scoring, str):
            score_dict.update({self.scoring + '_score': self.scoring})
        else:
            for item in self.scoring:
                score_dict.update({item + '_score': item})
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
        # print(score_dict)
        return score_dict

    def get_scoring_result(self, y, y_pred, y_prob, y_score=None):
        process_msg = ""
        if y_score is None:
            y_score = y_prob
        module_name = __import__("sklearn.metrics", fromlist='*')
        # print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
        score_dict = self._get_score_dict()
        # print(score_dict)
        # start get_scoring_result
        score_result_dict = {}
        for k, v in score_dict.items():
            score_func = getattr(module_name, k)
            sig = signature(score_func)
            y_flag = str(list(sig.parameters.keys())[1])
            if y_flag == 'y_pred':
                y_flag = y_pred
            elif y_flag == 'y_prob':
                y_flag = y_prob
            elif y_flag == 'y_score':
                y_flag = y_score
            else:
                raise ValueError("having new metrics that its 2nd param is not y_pred y_prob or y_score in sklearn !!!")
            if y_flag is None:
                raise ValueError(k, "%s is None !!!" % (y_flag))
            score_result = score_func(y, y_flag)
            # accuracy: (test=0.926)
            # print("%s: (test=%s)" % (v, score_result), end=" ")
            process_msg += "%s: (test=%.3f) " % (v, score_result)
            score_result_dict.update({v: score_result})
        return process_msg, score_result_dict

    def get_params_msg(self, params: dict):
        """
        format params:
        max_depth=3, min_child_weight=1;
        :param params:
        :return:
        """
        params_msg = ""
        for k, v in params.items():
            params_msg += "{}={}, ".format(k, v)

        params_msg = params_msg[: -2] + '; '
        # print("params_msg:", params_msg)
        return params_msg

    def get_cv_results(self):
        cv_results = {}
        cv_score_temp = {}
        # define attr of cv_results and cv_score_temp
        score_dict = self._get_score_dict()
        for k, v in score_dict.items():
            cv_results.update({"rank_test_%s" % v: [], "mean_test_%s" % v: [], "std_test_%s" % v: [], })
            cv_score_temp.update({v: np.array([])})
        cv_results.update({"params": []})

        # set data into cv_results
        n = self.cv
        for i in range(0, len(self.all_out), n):
            cv_out = self.all_out[i:i + n]
            cv_results["params"].append(cv_out[0][0])
            cv_out_score_temp = copy.deepcopy(cv_score_temp)
            for idx in range(0, self.cv):
                # set mean_test and std_test for per cv
                cv_out_score = cv_out[idx][1]
                for k, v in score_dict.items():
                    cv_out_score_temp[v] = np.append(cv_out_score_temp[v], cv_out_score[v])
            # print(cv_out_score_temp)
            for k, v in cv_out_score_temp.items():
                cv_results["mean_test_%s" % k].append(v.mean())
                # Weighted std is not directly available in numpy
                # array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2,
                #                                 axis=1, weights=weights))
                cv_results["std_test_%s" % k].append(v.std())
        # print("not rank:", cv_results)
        cv_results = self.rank_cv_result(cv_results)
        # print("ranked:", cv_results)
        return cv_results

    def rank_cv_result(self, cv_results):
        for item in self.scoring:
            # sorted by mean
            # obj = pd.Series(cv_results["mean_test_%s" % item])
            # c = obj.rank(ascending=False, method="min")
            # print(c.values.astype(int))
            # cv_results["rank_test_%s" % item].extend(c.values.astype(int))

            # sorted by mean and std
            df = pd.DataFrame({"mean": cv_results["mean_test_%s" % item], "std": cv_results["std_test_%s" % item]})
            cv_results["rank_test_%s" % item] = df.sort_values(by=['std', 'mean'])['mean'] \
                .rank(method='first', ascending=False).values.astype(int)
        return cv_results

    def get_best_estimator_params(self):
        if isinstance(self.refit, str):
            index = list(self.cv_results_["rank_test_%s" % self.refit]).index(1)
        return self.cv_results_["params"][index]

    def get_best_estimator(self):
        estimator = copy.deepcopy(self.estimator)
        estimator = self.set_estimator_params(estimator, self.best_estimator_params_)
        self.model_fit(estimator, train_X, train_y)
        return estimator


def performance_result(clf, test_X, test_y):
    print("Start prediction!!!")
    best_model = clf.best_estimator_
    # best_model = clf
    y_pred = best_model.predict(test_X)
    y_pred_prob_temp = best_model.predict_proba(test_X)
    y_pred_prob = []
    if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
        y_pred_prob = y_pred_prob_temp[:, 0]
    else:
        y_pred_prob = y_pred_prob_temp[:, 1]

    print("Performance evaluation!!!")
    auc = roc_auc_score(test_y, y_pred_prob)
    aupr = average_precision_score(test_y, y_pred_prob)
    acc = accuracy_score(test_y, y_pred)
    print("AUC : ", auc)
    print("AUPR : ", aupr)
    print("acc:", acc)

    p = 0  # 正确分类的个数
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(test_y)):  # 循环检测测试数据分类成功的个数
        if y_pred_prob[i] >= 0.5 and test_y[i] == 1:
            p += 1
            TP += 1
        elif y_pred_prob[i] < 0.5 and test_y[i] == 0:
            p += 1
            TN += 1
        elif y_pred_prob[i] < 0.5 and test_y[i] == 1:
            FN += 1
        elif y_pred_prob[i] >= 0.5 and test_y[i] == 0:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("total:", len(test_y), "\tacc num:", p, "\tTP:", TP, "\tTN:", TN, "\tFP:", FP, "\tFN:", FN)
    print("acc: ", p / len(test_y))  # 输出测试集准确率
    print("precision:", precision)
    print("recall:", recall)
    print("F1-score:", (2 * precision * recall) / (precision + recall))

    count = (y_pred == test_y).sum()
    print("count: ", count)
    print("acc: ", count / len(test_y))  # 输出测试集准确率


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
        'n_estimators': [2, 5, 8],
        'n_trees': [50, 100, 150, 200],
        'predictors': ['xgboost', 'lightgbm', 'forest'],
        'max_layers': [20, 50, 80],
        'use_predictor': [True]
    },
    {
        'n_estimators': [2, 5, 8],
        'n_trees':  [50, 100, 150, 200],
        'max_layers': [20, 50, 80]
    },
]
# parameters = [
#
#     {
#         'n_estimators': [2, 5],
#         'max_layers': [layer for layer in range(20, 40, 10)],
#         'predictors': ['xgboost', 'lightgbm', 'forest'],
#         'use_predictor': [True]
#     },
#
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
deep_forest = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest', verbose=0)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
cv_params = {'predictor': ['xgboost', 'lightgbm', 'forest']}

clf = MyGridSearchCV(train_X, train_y, deep_forest, parameters, met_grid, refit='roc_auc', n_jobs=3, cv=5)
writeRank2csv(met_grid, clf)
performance_result(clf, test_X, test_y)

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
