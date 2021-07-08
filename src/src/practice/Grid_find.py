# Title     : Grid_find
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/5/31

import random
import numpy as np
import os, sys, argparse
import time

from sklearn.model_selection import GridSearchCV

import mapping as mapping

map = mapping.map

BATCH_SIZE = 128
# 32 64  128 256 512 1024

import pandas as pd
from sklearn.svm import SVC, LinearSVC, NuSVC


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y


def lstm_train(X, y):
    print('网格寻优......')

    model = LinearSVC(penalty='l1', dual=False, tol=0.1, C=0.1, verbose=True, max_iter=3)

    tol = [1, 0.1, 0.01]
    C = [0.1, 0.05, 0.01, 0.05, 0.001]
    max_iter = [3, 4, 5, 8, 10, 20, 50]

    param_grid = dict(tol=tol, C=C, max_iter=max_iter)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=1)
    print("You get it?")
    grid_result = grid.fit(X, y)
    print("I get it!")
    print("grid search finished")

    # 总结结果
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    file = open('grid.txt', 'a')  # w:只写，文件已存在则清空，不存在则创建
    for mean, stdev, param in zip(means, stds, params):
        file.writelines('\t' + str(mean) + '\t' + str(stdev) + '\t' + str(param))
    file.close()


def funciton(benchmark_data):
    data_ben = np.array(pd.read_csv(benchmark_data))
    X = data_ben[:, 1:]
    y = data_ben[:, 0]

    X, y = shuffleData(X, y)

    lstm_train(X, y)
    print('=== SUCCESS ===')


def main():
    parser = argparse.ArgumentParser(description="deep learning 6mA analysis in rice genome")

    parser.add_argument("--benchmark_data", type=str, help="h_b_all.fa", required=True)
    args = parser.parse_args()

    DataCSV_ben = os.path.abspath(args.benchmark_data)

    if not os.path.exists(DataCSV_ben):
        print("The csv benchmark_data not exist! Error\n")
        sys.exit()

    funciton(DataCSV_ben)


if __name__ == "__main__":
    ts = time.time()
    main()
    print("training time: ", (time.time() - ts) / 60, "minutes")

#   python Grid_find.py --benchmark_data databases/benchmark_elmo/r_l_elmo.csv
