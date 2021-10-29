from sklearn.model_selection import train_test_split

a = [[1, 2], [2, 4], [3, 4], [4, 5], [1, 6], [4, 6]]
b = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.33, random_state=20, stratify=b)
print(X_train, y_train, )
print(X_test, y_test)
