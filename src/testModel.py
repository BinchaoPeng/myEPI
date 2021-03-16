from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch


def testModel(testLoader, module):
    y_pred = []
    y_test = []
    with torch.no_grad():
        for i, (x, y) in enumerate(testLoader, 1):
            pred = module(x)
            # correct += (pred.ge(0.5) == y).sum().item()
            # print("pred shape:", pred.shape)
            # print("pred:", pred)
            for item_pred, item_test in zip(pred.numpy(), y.numpy()):
                y_pred.append(item_pred)
                y_test.append(item_test)
        # percent = '%.2f' % (100 * correct / total)
        # print(f'Test set: Accuracy {correct}/{total} {percent}%')

        y_test = np.array(y_test).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        # print(type(y_pred))
        # print(type(y_test))
        # print("y_pred:", y_pred)
        # print("y_test:", y_test)
        auc = roc_auc_score(y_test, y_pred)
        aupr = average_precision_score(y_test, y_pred)

        print("test AUC : ", auc)
        print("test AUPR : ", aupr)
    return auc, aupr
