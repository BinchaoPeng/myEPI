import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools,sys
from utils import time_since



def trainModel(start, len_trainSet, epoch, trainLoader, module, criterion, optimal):
    total_loss = 0
    y_pred = []
    y_test = []
    for i, (x, y) in enumerate(trainLoader, 1):
        # print(trainLoader)
        # print(len(trainLoader))
        # print("train_x", x)
        # print("train_x", len(x[0]))
        # print("train_y", y)
        pred = module(x)
        # print(y_pred, y)

        loss = criterion(pred.type(torch.float), y.type(torch.float))
        optimal.zero_grad()
        loss.backward()
        optimal.step()

        total_loss += loss.item()
        y_pred.append(pred.tolist())
        y_test.append(y.tolist())
        if i % 5 == 0:
            # print("len(trainLoader):", len(trainLoader))
            # print("len(x)", len(x))
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(x[0])}/{len_trainSet}]', end='')
            print(f'loss = {total_loss / (i * len(x[0]))}')

    print(str(sys.getsizeof(y_pred)/1000), "KB")
    y_test = list(itertools.chain.from_iterable(y_test))
    y_pred = list(itertools.chain.from_iterable(y_pred))
    auc = roc_auc_score(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    # del y_test, y_pred

    print("train AUC : ", auc)
    print("train AUPR : ", aupr)
    return auc, aupr



# def getScore(y_pred_list=[], y_test_list=[]):
#
#
#
#     for item_pred, item_test in zip(pred, y):
#         y_pred.append(item_pred)
#         y_test.append(item_test)
#
#     y_test = np.array(y_test).reshape(-1)
#     y_pred = np.array(y_pred).reshape(-1)
#     auc = roc_auc_score(y_test, y_pred)
#     aupr = average_precision_score(y_test, y_pred)
#     # del y_test, y_pred
#
#     print("train AUC : ", auc)
#     print("train AUPR : ", aupr)
#     return auc, aupr