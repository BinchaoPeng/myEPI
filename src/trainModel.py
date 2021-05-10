import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools, sys
from utils import time_since


def create_tensor(tensor, USE_GPU=False):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def trainModel(num_iter, start_time, USE_GPU, len_trainSet, epoch, trainLoader, module, criterion, optimal):
    total_loss = 0
    y_pred = []
    y_test = []
    for i, (x, y) in enumerate(trainLoader, 1):
        # if USE_GPU:
        #     device = torch.device("cuda:0")
        #     x.to(device)
        #     y.to(device)
        # print(trainLoader)
        # print(len(trainLoader))
        # print("train_x", x)
        # print("train_x", len(x[0]))
        # print("train_y", y)
        pred = module(x)
        # print(y_pred, y)
        # print(type(y))
        # print(type(pred))
        # loss = criterion(pred.type(torch.cuda.float), y.type(torch.float))
        loss = criterion(pred.cpu().type(torch.float), y.cpu().type(torch.float))
        # loss = criterion(torch.FloatTensor(pred.cpu()), torch.FloatTensor(y.cpu()))
        # loss = criterion(pred, y)
        optimal.zero_grad()
        loss.backward()
        optimal.step()

        total_loss += loss.item()
        y_pred.append(pred.tolist())
        y_test.append(y.tolist())
        if i % num_iter == 0:
            # print("len(trainLoader):", len(trainLoader))
            # print("len(x)", len(x))
            print(f'[{time_since(start_time)}] Epoch {epoch} ', end='')
            # print(f'[{i * len(x[0])}/{len_trainSet}]', end='')
            # print(f'loss = {total_loss / (i * len(x[0]))}')
            print(f'[{i * len(y)}/{len_trainSet}]', end='')
            print(f'loss = {total_loss / (i * len(y))}')

    # print(str(sys.getsizeof(y_pred) / 1000), "KB")
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
