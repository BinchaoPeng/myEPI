import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import math
from sklearnex import patch_sklearn

patch_sklearn()
"""
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm停止训练的误差精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；默认为200MB。
（10）class_weight: 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.如果给定参数‘balanced’，则使用y的值自动调整与输入数据中的类频率成反比的权重。缺省的话自适应；
（11）verbose: 是否启用详细输出。 此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
 ps：7,8,9一般不考虑。
 主要调节的参数有：C、kernel、degree、gamma、coef0。
 ★fit()方法：用于训练SVM，具体参数已经在定义SVC对象的时候给出了，这时候只需要给出数据集X和X对应的标签y即可。
 ★predict()方法：基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示个测试样本的类别。
"""

"""
cell and feature choose
"""
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[7]
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

"""
read X,y
X = [[2, 3, 7, 8],
     [4, 6, 8, 9],
     [7, 6, 4, 2]
     ]

y = [1, 0, 1]

test_dataset = [[2, 5, 6, 2],
                [4, 9, 0, 3],
                [7, 2, 7, 2]
                ]

test_labels = [
    [1],
    [1],
    [0]
]
"""

parameters = [
    {
        'C': [math.pow(2, i) for i in range(-5, 5)],
        'gamma': [math.pow(2, i) for i in range(-5, 5)],
        'kernel': ['rbf']
    },
    {
        'C': [math.pow(2, i) for i in range(-5, 5)],
        'kernel': ['linear', 'poly', 'sigmoid']
    },
    {
        'C': [math.pow(2, i) for i in range(-5, 5)],
        'degree': [2, 3, 4, 5, 6],
        'kernel': ['poly']
    }
]

svc = SVC(probability=True, )  # 调参
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
clf = GridSearchCV(svc, parameters, cv=3, n_jobs=15, scoring=met_grid, refit='roc_auc', verbose=3)
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
