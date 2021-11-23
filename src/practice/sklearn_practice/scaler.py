from sklearn import preprocessing
import numpy as np

"""
把训练集和测试集放在一块进行标准化处理，求出共同的均值和方差，然后X减去均值再除以方差。处理成均值为0， 方差为1的数据。 参考了测试集本身。
"""
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

print(X_train.mean(axis=0))
print(X_train.std(axis=0))
print((X_train - X_train.mean(axis=0)))
print((X_train - X_train.mean(axis=0)) / X_train.std(axis=0))

X_scaled = preprocessing.scale(X_train)
print(X_scaled)
# array([[0., -1.22, 1.33],
#        [1.22, 0., -0.26],
#        [-1.22, 1.22, -1.06]])

# 处理后数据的均值和方差
print(X_scaled.mean(axis=0))
# array([0., 0., 0.])

print(X_scaled.std(axis=0))
# array([1., 1., 1.])

print("========================StandardScaler============================")
"""
只求出训练集的均值和方差，然后让训练集和测试集的X都减去这个均值然后除以方差。 均值和方差的计算没有参考测试集。
"""
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
X_test = np.array([[-1., 3., 0.], [2., 1., 1.]])
print(X_train.mean(axis=0))
print(X_train.std(axis=0))

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
print(scaler)
print(scaler.mean_)
print(scaler.var_)  # variance
print(scaler.scale_)  # equal std
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

# 可以直接使用训练集对测试集数据进行转换
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

print((X_test - X_train.mean(axis=0)) / X_train.std(axis=0))
