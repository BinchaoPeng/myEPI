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

# -*- coding:utf-8 -*-
'''
Stacking方法
'''
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings;

warnings.filterwarnings(action='ignore')

# ========================================================
#  载入iris数据集
# ========================================================

iris = load_iris()
X = iris.data[:, :5]
y = iris.target

print('feature=', X)
print('target=', y)


# ========================================================
#  实现Stacking集成
# ========================================================

def StackingMethod(X, y):
    '''
    Stacking方法实现分类
    INPUT -> 特征, 分类标签
    '''
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(X)  # 训练标准化对象
    traffic_feature = scaler.transform(X)  # 转换数据集
    feature_train, feature_test, target_train, target_test = model_selection.train_test_split(X, y, test_size=0.3,
                                                                                              random_state=0)

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              # use_probas=True, 类别概率值作为meta-classfier的输入
                              # average_probas=False,  是否对每一个类别产生的概率值做平均
                              meta_classifier=LogisticRegression())

    sclf.fit(feature_train, target_train)

    # 模型测试
    predict_results = sclf.predict(feature_test)
    print(accuracy_score(predict_results, target_test))
    conf_mat = confusion_matrix(target_test, predict_results)
    print(conf_mat)
    print(classification_report(target_test, predict_results))

    # 5折交叉验证
    for clf, label in zip([clf1, clf2, clf3, sclf],
                          ['Logistic Regression', 'Random Forest', 'naive Bayes', 'StackingModel']):
        scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    return sclf


# ========================================================
#  主程序
# ========================================================

if __name__ == '__main__':
    model = StackingMethod(X, y)
