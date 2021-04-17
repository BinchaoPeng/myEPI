import itertools

import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def testModel(testLoader, module):
    y_pred = []
    y_test = []
    with torch.no_grad():
        for i, (x, y) in enumerate(testLoader, 1):
            pred = module(x)

            y_pred.append(pred.tolist())
            y_test.append(y.tolist())
        y_test = list(itertools.chain.from_iterable(y_test))
        y_pred = list(itertools.chain.from_iterable(y_pred))

        # print(type(y_pred))
        # print(type(y_test))
        print("y_pred:", y_pred)
        print("y_test:", y_test)
        auc = roc_auc_score(y_test, y_pred)
        aupr = average_precision_score(y_test, y_pred)

        print("test AUC : ", auc)
        print("test AUPR : ", aupr)
    return auc, aupr
