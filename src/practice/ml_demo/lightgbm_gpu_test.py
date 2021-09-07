import lightgbm
import numpy as np


def check_gpu_support():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    print(label)
    train_data = lightgbm.Dataset(data, label=label)
    params = {'num_iterations': 1, 'device': 'gpu'}
    try:
        gbm = lightgbm.train(params, train_set=train_data)
        print("GPU True !!!")
    except Exception as e:
        print("GPU False !!!")


if __name__ == '__main__':
    check_gpu_support()
