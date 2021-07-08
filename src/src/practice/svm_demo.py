from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import time, math
from sklearn.model_selection import GridSearchCV

"""
step_1
dataset
"""
X = [[2, 3, 7, 8],
     [4, 6, 8, 9],
     [7, 6, 4, 2],
     [4, 6, 2, 8],
     [1, 3, 6, 9],
     [4, 3, 5, 2]
     ]

y = [1, 0, 1, 0, 1, 1]

test_X = [[2, 5, 6, 2],
          [4, 9, 0, 3],
          [7, 2, 7, 2]
          ]

test_y = np.array([
    [1],
    [1],
    [0]
])

"""
step_2
hyper params
"""
parameters = [
    {
        'C': [math.pow(2, i) for i in range(-10, 10)],
        'gamma': [math.pow(2, i) for i in range(-10, 10)],
        'kernel': ['rbf']
    },
    {
        'C': [math.pow(2, i) for i in range(-10, 10)],
        'kernel': ['linear', 'poly', 'sigmoid']
    }
]

"""
step_3
grid search
"""
svc = SVC(probability=True, )  # 调参
met_grid = ['f1', 'roc_auc']
clf = GridSearchCV(svc, parameters, cv=3, n_jobs=12, scoring=met_grid, refit='roc_auc', verbose=4)
grid_result = clf.fit(X, y)
print("have found the BEST param!!!")

# 总结结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# file = open('grid.txt', 'a')  # w:只写，文件已存在则清空，不存在则创建
# for mean, stdev, param in zip(means, stds, params):
#     file.writelines('\t' + str(mean) + '\t' + str(stdev) + '\t' + str(param))
# file.close()

"""
grid.scores：给出不同参数组合下的评价结果
grid.best_params_：最佳结果的参数组合
grid.best_score_：最佳评价得分
grid.best_estimator_：最佳模型
"""
print("best_params:", clf.best_params_)
best_model = clf.best_estimator_

y_pred = best_model.predict(test_X)
y_pred_prob_temp = best_model.predict_proba(test_X)
y_pred_prob = y_pred_prob_temp[:, 1]

auc = roc_auc_score(test_y.flatten(), y_pred_prob)
aupr = average_precision_score(test_y.flatten(), y_pred_prob)
print("AUC : ", auc)
print("AUPR : ", aupr)
print(roc_auc_score([1, 1, 0], [0.54162502, 0.54186845, 0.54159902]))
print(average_precision_score([1, 1, 0], [0.54162502, 0.54186845, 0.54159902]))

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

count = (y_pred == test_y.flatten()).sum()
print("count: ", count)
print("acc: ", count / len(test_y))  # 输出测试集准确率
