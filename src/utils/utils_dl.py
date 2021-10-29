import time, math
import random

import numpy as np
import torch


def time_since(start):
    s = time.time() - start
    # s = 62 - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def use_gpu_first():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_gpu = True if device in 'cuda' else False
    return device, use_gpu


def time_since_test(start):
    s = time.time() - start
    # s = 62 - start
    if s < 60:
        return '%.1fs' % s
    elif 60 < s and s < 3600:
        s = s / 60
        return '%.1fmin' % s
    else:
        m = math.floor(s / 60)
        s -= m * 60
        h = math.floor(m / 60)
        m -= h * 60
        return '%dh %dm %ds' % (h, m, s)


if __name__ == '__main__':
    print(time_since_test(1))
    print(time.time())
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3, ], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    X, y = shuffleData(X, y, seed=1)
    print(X, y)
