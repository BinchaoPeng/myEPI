from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import time

X = [[2, 3, 7, 8],
     [4, 6, 8, 9],
     [7, 6, 4, 2]
     ]

y = [1, 0, 1]

test_X = [[2, 5, 6, 2],
          [4, 9, 0, 3],
          [7, 2, 7, 2]
          ]

test_y = np.array([
    [1],
    [1],
    [0]
])

clf = SVC(kernel='rbf', probability=True)  # 调参
clf.fit(X, y)  # 训练
print(clf.fit(X, y))  # 输出参数设置

y_pred = clf.predict(test_X)
y_pred_prob = clf.predict_proba(test_X)
y_pred_prob1 = y_pred_prob[:, 0]

auc = roc_auc_score(test_y.flatten(), y_pred_prob1)
aupr = average_precision_score(test_y.flatten(), y_pred_prob1)
print("AUC : ", auc)
print("AUPR : ", aupr)

p = 0  # 正确分类的个数
for i in range(len(test_y)):  # 循环检测测试数据分类成功的个数
    if y_pred[i] >= 0.5 and test_y[i][0] == 1:
        p += 1
print(len(test_y))
print(p / len(test_y))  # 输出测试集准确率

p = 0
for i in range(len(test_y)):  # 循环检测测试数据分类成功的个数
    if y_pred[i] == test_y[i][0]:
        p += 1
print(p / len(test_y))  # 输出测试集准确率
