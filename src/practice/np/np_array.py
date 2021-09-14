import numpy as np

a = [1, 5, 1, 3, 2, 5]

print(a.index(1))

b = np.array([2])

print(b.mean())
print(b.std())

b = np.array([[1, 4, 3], [4, 7, 6]])
print(b.flatten())

X_en1 = [1, 2]
X_pr1 = [5, 6, 7, 8]
a, b = np.array(X_en1), np.array(X_pr1)
test_X = [np.hstack((item1, item2)) for item1, item2 in zip(a, b)]
print(test_X)
