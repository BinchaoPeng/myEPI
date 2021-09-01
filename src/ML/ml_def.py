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
    confusion_matrix, plot_confusion_matrix, recall_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def get_data_np_dict(cell_name, feature_name, method_name):
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
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def get_score_dict(scoring):
    score_dict = {}
    if isinstance(scoring, str):
        score_dict.update({scoring + '_score': scoring})
    else:
        for item in scoring:
            score_dict.update({item + '_score': item})
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
    # print(score_dict)
    return score_dict


def get_scoring_result(scoring, y, y_pred, y_prob, y_score=None, is_base_score=True):
    process_msg = ""
    if y_score is None:
        y_score = y_prob
    module_name = __import__("sklearn.metrics", fromlist='*')
    # print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
    score_dict = get_score_dict(scoring)
    # print(score_dict)
    # start get_scoring_result
    score_result_dict = {}
    if is_base_score:
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        score_result_dict.update({"total": len(y), "TP": TP, "TN": TN, "FP": FP, "FN": FN, "precision": precision,
                                  "recall": recall})
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


def writeCVRank2csv(met_grid, clf, dir_name, index=None):
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

    ex_dir_name = '%s_%s_%s' % (feature_name, method_name, dir_name)
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


def writeRank2csv(met_grid, clf, dir_name, index=None):
    print("write rank test to csv!!!")
    csv_rows_list = []
    header = []
    for m in met_grid:
        rank_test_score = 'rank_test_' + m
        header.append(rank_test_score)
        csv_rows_list.append(clf.cv_results_[rank_test_score])

    csv_rows_list.append(clf.cv_results_['params'])
    header.append('params')
    results = list(zip(*csv_rows_list))
    print("write over!!!")

    ex_dir_name = '%s_%s_%s' % (feature_name, method_name, dir_name)
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
            idx = list(self.cv_results_["rank_test_%s" % self.refit]).index(1)
        return self.cv_results_["params"][idx]

    def get_best_estimator(self, train_X, train_y):
        estimator = copy.deepcopy(self.estimator)
        estimator = self.set_estimator_params(estimator, self.best_estimator_params_)
        self.model_fit(estimator, train_X, train_y)
        return estimator


class RunAndScore:

    def __init__(self, data_list_dict, estimator, parameters, scoring, refit, n_jobs, ):
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.scoring = scoring
        self.refit = refit
        self.data_list_dict = data_list_dict
        self.candidate_params = self.get_candidate_params(parameters)

        self.all_out = self.run_and_score()
        self.cv_results_ = self.get_cv_results()
        self.best_estimator_params_ = self.get_best_estimator_params()
        self.best_estimator_ = self.get_best_estimator()

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

    def get_candidate_params(self, parameters):
        """
        return params_dict_list
        :param parameters:
        :return:
        """
        return list(ParameterGrid(parameters))

    def fit_and_predict(self, cand_idx, params):
        n_candidates = len(self.candidate_params)
        process_msg = "[fit {}/{}] END ".format(cand_idx, n_candidates)
        start = time.time()
        deep_forest = copy.deepcopy(self.estimator)
        model = self.set_estimator_params(deep_forest, params)
        model.fit(self.data_list_dict["train_X"], self.data_list_dict["train_y"])
        y_pred = model.predict(self.data_list_dict["test_X"])
        y_pred_prob_temp = model.predict_proba(self.data_list_dict["test_X"])
        y_pred_prob = []
        if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
            y_pred_prob = y_pred_prob_temp[:, 0]
        else:
            y_pred_prob = y_pred_prob_temp[:, 1]

        params_msg = ""
        for k, v in params.items():
            params_msg += "{}={}, ".format(k, v)

        process_msg += params_msg[: -2] + '; '

        [score_result_msg, score_result_dict] = get_scoring_result(self.scoring, self.data_list_dict["test_y"], y_pred,
                                                                   y_pred_prob)
        process_msg += score_result_msg
        process_msg += time_since(start)
        print(process_msg)
        print([params, score_result_dict])
        return [params, score_result_dict]

    def run_and_score(self):
        """
        all_out = [out0,out1,out2,...]
        out = [[params, score_result_dict],[params, score_result_dict],...]
        score_result_dict = {"score0":s0,"score1":s1,...}
        :return:
        """
        n_candidates = len(self.candidate_params)
        print("Fitting, totalling {0} fits".format(n_candidates))
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.fit_and_predict)
                           (cand_idx, params)
                           for cand_idx, params in enumerate(self.candidate_params, 1))
            # print(out)
            n_candidates = len(self.candidate_params)
            if len(out) < 1:
                raise ValueError('No fits were performed. '
                                 'Was the CV iterator empty? '
                                 'Were there no candidates?')
            elif len(out) != n_candidates:
                raise ValueError('cv.split and cv.get_n_splits returned '
                                 'inconsistent results. Expected {} '
                                 'splits, got {}'
                                 .format(n_candidates, len(out)))
            print("score_result_dict:", out)
            all_out.extend(out)
        return all_out

    def _get_score_dict(self):
        """
        score_dict: {"xxx_score":score,...}
        :return:
        """
        score_dict = {}
        if isinstance(self.scoring, str):
            score_dict.update({self.scoring + '_score': self.scoring})
        else:
            for item in self.scoring:
                score_dict.update({item + '_score': item})
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
        # print(score_dict)
        return score_dict

    def get_cv_results(self):
        cv_results = {}
        cv_score_temp = {}
        # define attr of cv_results and cv_score_temp
        score_dict = self._get_score_dict()
        for k, v in score_dict.items():
            cv_results.update({"rank_test_%s" % v: []})
            cv_results.update({v: np.array([])})
        cv_results.update({"params": []})

        # set data into cv_results
        for cv_out in self.all_out:
            cv_results["params"].append(cv_out[0])
            cv_out_score = cv_out[1]
            for k, v in score_dict.items():
                cv_results[v] = np.append(cv_results[v], cv_out_score[v])
                # cv_results[v].append(cv_out_score[v])
        # print("not rank:", cv_results)
        cv_results = self.rank_cv_result(cv_results)
        # print("ranked:", cv_results)
        return cv_results

    def rank_cv_result(self, cv_results):
        for item in self.scoring:
            # sorted by mean
            obj = pd.Series(cv_results[item])
            c = obj.rank(ascending=False, method="min")
            print(c.values.astype(int))
            cv_results.update({"rank_test_%s" % item: c.values.astype(int)})

            # # sorted by mean and std
            # df = pd.DataFrame({"mean": cv_results["mean_test_%s" % item], "std": cv_results["std_test_%s" % item]})
            # cv_results["rank_test_%s" % item] = df.sort_values(by=['std', 'mean'])['mean'] \
            #     .rank(method='first', ascending=False).values.astype(int)
        return cv_results

    def get_best_estimator_params(self):
        if isinstance(self.refit, str):
            idx = list(self.cv_results_["rank_test_%s" % self.refit]).index(1)
        return self.cv_results_["params"][idx]

    def get_best_estimator(self):
        estimator = copy.deepcopy(self.estimator)
        estimator = self.set_estimator_params(estimator, self.best_estimator_params_)
        self.model_fit(estimator, self.data_list_dict["train_X"], self.data_list_dict["train_y"])
        return estimator


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
        'n_trees': [50, 100, 150, 200],
        'max_layers': [20, 50, 80]
    },
]
parameters = [

    {
        'n_estimators': [2, 4],
        # 'max_layers': [layer for layer in range(20, 40, 10)],
        'predictors': ['xgboost', 'lightgbm', 'forest'],
        'use_predictor': [True]
    },

]
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

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[1]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest']
method_name = method_names[2]
data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

deep_forest = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest', verbose=0)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']

clf = RunAndScore(data_list_dict, deep_forest, parameters, met_grid, refit="roc_auc", n_jobs=2)
writeRank2csv(met_grid, clf, "fit_and_score")
