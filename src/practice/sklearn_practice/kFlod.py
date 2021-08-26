import numpy as np
from sklearn.model_selection import StratifiedKFold

X = ["a", "b", "c", "d", "e", 'f', 'g', 'h', 'i', 'j']
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
X = np.array(X)  # numpy
kf = StratifiedKFold(n_splits=3)
for index, item in enumerate(kf.split(X, y), 1):
    print(item)
    train, test = item
    print(index)
    X_train, X_test = X[train], X[test]
    print(X_train, X_test)

