from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 导入IRIS数据集
iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)

# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
rfe = RFE(estimator=LogisticRegression(max_iter=1000), step=1)
rfe.fit(iris.data, iris.target)
print(rfe.get_support(indices=True))
print(rfe.support_)
print(rfe.ranking_)
# print(rfe.n_features_)
# print(rfe.transform(iris.data))
