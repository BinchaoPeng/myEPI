import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import time
from utils import time_since, use_gpu_first

device, USE_GPU = use_gpu_first()


def create_tensor(tensor, USE_GPU=False):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def trainModel(num_iter, start_time, len_trainSet, epoch, trainLoader, model, criterion, optimal, scheduler=None):
    scaler = torch.cuda.amp.GradScaler() # 混合精度加速训练
    total_loss = 0
    y_pred = []
    y_test = []
    time0 = time.time()
    for i, (x, y) in enumerate(trainLoader, 1):
        y = y.to(device=device, dtype=torch.float32)

        # print(trainLoader)
        # print(len(trainLoader))
        # print("train_x", x)
        # print("train_x", len(x[0]))
        # print("train_y", y)
        optimal.zero_grad()

        with torch.cuda.amp.autocast():  # 混合精度加速训练
            time1 = time.time()
            pred = model(x)
            print("=model(x) time:", time.time() - time1)
        # print(y_pred, y)
        # print(type(y))
        # print(type(pred))
        # print(y.device)
        # print(pred.device)
        # print(y.dtype)
        # print(pred.dtype)
        # loss = criterion(pred.cpu().type(torch.float), y.cpu().type(torch.float))
            time2 = time.time()
            loss = criterion(pred, y)
        # loss.backward()
        # optimal.step()
        # # 混合精度加速训练
        scaler.scale(loss).backward()
        scaler.step(optimal)
        scaler.update()
        print("=loss and optimal time:", time.time() - time2, "\n")
        total_loss += loss.item()
        y_pred.append(pred.tolist())
        y_test.append(y.tolist())
        if i % num_iter == 0:
            # # update lr
            # scheduler.step(total_loss)

            # print("len(trainLoader):", len(trainLoader))
            # print("len(x)", len(x))
            print(f'[{time_since(start_time)}] Epoch {epoch} ', end='')
            # print(f'[{i * len(x[0])}/{len_trainSet}]', end='')
            # print(f'loss = {total_loss / (i * len(x[0]))}')
            print(f'[{i * len(y)}/{len_trainSet}]', end='')
            print(f'loss = {total_loss / (i * len(y))}')

    print("===train a epoch time:", time.time() - time0)

    time4 = time.time()
    # print(str(sys.getsizeof(y_pred) / 1000), "KB")
    y_test = list(itertools.chain.from_iterable(y_test))
    y_pred = list(itertools.chain.from_iterable(y_pred))
    auc = roc_auc_score(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    # del y_test, y_pred

    print("train AUC : ", auc)
    print("train AUPR : ", aupr)
    print("train an epoch's metrics time:", time.time() - time4)
    return auc, aupr

