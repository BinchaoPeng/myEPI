import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d", "e", 'f', 'g', 'h', 'i', 'j']
X = np.array(X)  # numpy
kf = KFold(n_splits=3)
for train, test in kf.split(X):
    print(train, test)
    X_train, X_test = X[train], X[test]
    print(X_train, X_test)

