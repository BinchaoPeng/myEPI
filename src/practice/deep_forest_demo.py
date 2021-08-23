import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def performance_result(clf):
    print("Start prediction!!!")
    # best_model = clf.best_estimator_
    best_model = clf
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


"""
cell and feature choose
"""
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[2]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost']
method_name = method_names[1]
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

n_estimators = [2, 3, 4, 5, 6, 7, 8, 9]
n_trees = [x for x in range(50, 550, 50)]
predictors = ['xgboost', 'lightgbm', 'forest']
max_layers = [layer for layer in range(20, 110, 10)]
for n_estimator in n_estimators:
    for n_tree in n_trees:
        for max_layer in max_layers:
            for use_predictor in (True, False):
                if use_predictor is True:
                    for predictor in predictors:
                        pass
                else:
                    pass

# model = CascadeForestClassifier(use_predictor=True, random_state=1, n_jobs=5, predictor='forest')
model = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest')
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
cv_params = {'predictor': ['xgboost', 'lightgbm', 'forest']}
cv_params = {'predictor': ['xgboost', 'forest']}
# clf = GridSearchCV(estimator=model, param_grid=cv_params, scoring=met_grid, refit='roc_auc', cv=5, n_jobs=1,
#                    verbose=3)
clf = cross_val_score(model, train_X, train_y, scoring='roc_auc', cv=5, verbose=3)
clf.fit(train_X, train_y)
# print('参数的最佳取值：{0}'.format(grid.best_params_))
# print('最佳模型得分:{0}'.format(grid.best_score_))
performance_result(clf)

"""
param doc
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
