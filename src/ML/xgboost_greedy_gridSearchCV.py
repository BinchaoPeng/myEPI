import os
import sys
import csv
import time
import warnings

start_time = time.time()

warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[3]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm']
method_name = method_names[3]
print("experiment: %s %s_%s" % (cell_name, feature_name, method_name))
dir_name = "run_and_score"

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

other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                'use_label_encoder': False, 'eval_metric': 'logloss', 'tree_method': 'gpu_hist'}


def xgboost_grid_greedy(cv_params, other_params, index):
    model = XGBClassifier(**other_params)
    met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
    clf = GridSearchCV(estimator=model, param_grid=cv_params, scoring=met_grid, refit='roc_auc', cv=5, n_jobs=3,
                       verbose=3)

    clf.fit(train_X, train_y)
    print('参数的最佳取值：{0}'.format(clf.best_params_))
    print('最佳模型得分:{0}'.format(clf.best_score_))
    # 参数的最佳取值：{'n_estimators': 50}
    # 最佳模型得分:0.6725274725274726
    writeRank2csv(met_grid, clf, index)
    performance_result(clf)
    return clf.best_params_


def writeRank2csv(met_grid, clf, index):
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

    ex_dir_name = 'pseknc_xgboost_5flod_grid'
    file_name = r'../../ex/%s/rank/%s_%s_xgboost_rank_%s.csv' % (ex_dir_name, cell_name, feature_name, index)
    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(results)
        f.close()


def performance_result(clf):
    print("Start prediction!!!")
    best_model = clf.best_estimator_
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


# 第一次：决策树的最佳数量也就是估计器的数目
print("第一次")
cv_params = {'n_estimators': list(range(50, 1050, 50))}
# cv_params = {'n_estimators': list(range(50, 300, 50))}
best_params = xgboost_grid_greedy(cv_params, other_params, '1')
other_params.update(best_params)

# 第二次
print("第二次")
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
# cv_params = {'max_depth': [3, 4, ], 'min_child_weight': [1, 2, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '2')
other_params.update(best_params)
# print(other_params)

# 第三次
print("第三次")
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
# cv_params = {'gamma': [0.1, 0.2, 0.3,]}
best_params = xgboost_grid_greedy(cv_params, other_params, '3')
other_params.update(best_params)
# print(other_params)

# 第四次
print("第四次")
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
# cv_params = {'subsample': [0.6, 0.7, ], 'colsample_bytree': [0.6, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '4')
other_params.update(best_params)
# print(other_params)

# 第五次
print("第五次")
cv_params = {'reg_alpha': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3],
             'reg_lambda': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3]}
# cv_params = {'reg_alpha': [0.05, ], 'reg_lambda': [0.05, 0.1, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '5')
other_params.update(best_params)
# print(other_params)

# 第六次
print("第六次")
cv_params = {'learning_rate': [0.001, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.75, 1.0]}
# cv_params = {'learning_rate': [0.01, 0.05, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '6')
other_params.update(best_params)
# print(other_params)
