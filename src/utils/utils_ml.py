import random


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
