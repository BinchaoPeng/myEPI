import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from testModel import testModel
from trainModel import trainModel, time_since
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from EPIDataset import EPIDataset
from modelBase import EPINet

"""
Hyper parameter
"""
N_EPOCHS = 15
batch_size = 128
num_works = 0
lr = 0.001
names = ['PBC', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
name = names[4]

np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽

trainSet = EPIDataset(name)
len_trainSet = len(trainSet)
print("trainSet data len:", len(trainSet))
trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=num_works)
print("trainLoader len:", len(trainLoader))

testSet = EPIDataset(name, is_train_set=False)
testLoader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=num_works, drop_last=True)

module = EPINet()
# print(module.parameters())
"""
loss and optimizer
"""
criterion = nn.BCELoss(reduction='sum')
optimal = optim.Adam(module.parameters(), lr=lr)

if __name__ == '__main__':

    # if USE_GPU:
    #     device = torch.device("cuda:0")
    #     module.to(device)

    start = time.time()
    acc_list = []
    test_auc_list = []
    test_aupr_list = []

    train_auc_list = []
    train_aupr_list = []
    for epoch in range(1, N_EPOCHS + 1):
        auc, aupr = trainModel(start, len_trainSet, epoch, trainLoader, module, criterion, optimal)
        train_auc_list.append(auc)
        train_aupr_list.append(aupr)
        print(f"============================[{time_since(start)}]train: EPOCH {epoch} is over!================")
        auc, aupr = testModel(testLoader, module)
        print(f"============================[{time_since(start)}]test: EPOCH {epoch} is over!================")
        test_auc_list.append(auc)
        test_aupr_list.append(aupr)
        # print("============================test: ACC is", acc, "=======================================")
        # torch.save(module, r'..\model\%sModule-%s.pkl' % (name, str(epoch)))
        torch.save(module, r'../model/model-%s.pkl' % (str(epoch)))  # must use /
        print("============================saved model !", "=======================================")
    # polt
    x = range(1, N_EPOCHS + 1)
    plt.plot(x, test_auc_list, 'b-o', label="test_auc")
    plt.plot(x, train_auc_list, 'r-o', label="train_auc")
    plt.ylabel("auc")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(x, test_aupr_list, 'b-o', label="test_aupr")
    plt.plot(x, train_aupr_list, 'r-o', label="train_aupr")
    plt.ylabel("aupr")
    plt.xlabel("epoch")
    plt.show()
