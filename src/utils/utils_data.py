import random
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import scale, StandardScaler


def shuffleData(X, y, seed=None):
    random.seed(seed)
    index = [i for i in range(len(X))]
    random.shuffle(index)
    # print(index)
    X = X[index]
    y = y[index]
    return X, y


def shuffle_data_list_dict(data_list_dict: dict, seed=None):
    train_X, train_y = shuffleData(data_list_dict["train_X"], data_list_dict["train_y"], seed)
    test_X, test_y = shuffleData(data_list_dict["test_X"], data_list_dict["test_y"], seed)
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def scaler(data_list_dict: dict, ):
    train_X = scale(data_list_dict["train_X"])
    test_X = scale(data_list_dict["test_X"])
    train_y = data_list_dict["train_y"]
    test_y = data_list_dict["test_y"]
    print("data Scaler!")
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def standardScaler(data_list_dict: dict):
    scaler = StandardScaler()
    scaler.fit(data_list_dict["train_X"])
    train_X = scaler.transform(data_list_dict["train_X"])
    test_X = scaler.transform(data_list_dict["test_X"])
    train_y = data_list_dict["train_y"]
    test_y = data_list_dict["test_y"]
    print("data standardScaler!")
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def feature_select_RFE(data_list_dict: dict, estimator):
    # rfe = RFE(estimator=lightgbm.sklearn.LGBMClassifier(device='gpu'), step=1)
    rfe = RFE(estimator=estimator, step=1)
    rfe.fit(data_list_dict["train_X"], data_list_dict["train_y"])
    print("RFE n_features:", rfe.n_features_)
    data_list_dict.update({"train_X": rfe.transform(data_list_dict["train_X"])})
    data_list_dict.update({"test_X": rfe.transform(data_list_dict["test_X"])})
    return data_list_dict


if __name__ == '__main__':
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3, ], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    X, y = shuffleData(X, y, seed=1)
    print(X, y)
