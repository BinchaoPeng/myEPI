import time, math
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


def end_train(max_aupr, max_auc, aupr, auc, count):
    if max_aupr < aupr:
        max_aupr = aupr
    if max_auc < auc:
        max_auc = auc

        max_aupr, max_auc = 0


if __name__ == '__main__':
    print(time_since(2))
