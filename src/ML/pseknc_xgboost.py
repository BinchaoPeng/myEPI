import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import math, time
from xgboost import XGBClassifier

"""
cell and feature choose
"""
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[5]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]

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
# print(type(self.X))
test_y = test_data[test_data.files[2]]
print("testSet len:", len(test_y))

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

xgboost = XGBClassifier(learning_rate=0.01,
                        n_estimators=10,  # 树的个数-10棵树建立xgboost
                        max_depth=4,  # 树的深度
                        min_child_weight=1,  # 叶子节点最小权重
                        gamma=0.,  # 惩罚项中叶子结点个数前的参数
                        subsample=1,  # 所有样本建立决策树
                        colsample_btree=1,  # 所有特征建立决策树
                        scale_pos_weight=1,  # 解决样本个数不平衡的问题
                        random_state=27,  # 随机数
                        slient=0,
                        )

parameters = [
    {
        'C': [math.pow(2, i) for i in range(-10, 10)],
        'gamma': [math.pow(2, i) for i in range(-10, 10)],
        'kernel': ['rbf']
    },
    {
        'C': [math.pow(2, i) for i in range(-10, 10)],
        'kernel': ['linear', 'poly', 'sigmoid']
    },
    {
        'C': [16],
        'degree': [2, 3, 4, 5, 6],
        'kernel': ['poly']
    }
]

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
clf = GridSearchCV(xgboost, parameters, cv=5, n_jobs=10, scoring=met_grid, refit='roc_auc', verbose=3)
print("Start Fit!!!")
clf.fit(train_X, train_y)
print("Found the BEST param!!!")
"""
grid.scores：给出不同参数组合下的评价结果
grid.best_params_：最佳结果的参数组合
grid.best_score_：最佳评价得分
grid.best_estimator_：最佳模型
"""
print("best_params:", clf.best_params_)
best_model = clf.best_estimator_

print("Start prediction!!!")
y_pred = best_model.predict(test_X)
y_pred_prob_temp = best_model.predict_proba(test_X)
y_pred_prob = y_pred_prob_temp[:, 1]

print("Performance evaluation!!!")
auc = roc_auc_score(test_y, y_pred_prob)
aupr = average_precision_score(test_y, y_pred_prob)
acc = accuracy_score(test_y, y_pred_prob)
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
