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
import model.modelBase as modelBase
import model.model_gru3 as model_gru3
import model.model_transformer as model_transformer
import model.model_transformer_1 as model_transformer_1
import model.model_transformer_2 as model_transformer_2
import model.model_pseknc_1 as model_pseknc_1
import model.modelBase as modelBase

"""
Hyper parameter
"""
N_EPOCHS = 40
batch_size = 8
num_works = 0
lr = 0.00001
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
name = names[1]

np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽

trainSet = EPIDataset(name, feature_name="")
len_trainSet = len(trainSet)
print("trainSet data len:", len(trainSet))
trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=num_works)
print("trainLoader len:", len(trainLoader))

testSet = EPIDataset(name, is_train_set=False)
len_testSet = len(testSet)
testLoader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=num_works, drop_last=True)

"""
base
"""
module = modelBase.EPINet()
# module = model_pseknc_1.EPINet()
# module = model_transformer_2.EPINet()
# module = model_transformer_1.EPINet()
# module = model_transformer.EPINet()
# module = model_gru3.EPINet()
# module = model_transformer.EPINet()


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
        auc, aupr = trainModel(20, start, len_trainSet, epoch, trainLoader, module, criterion, optimal)
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
    x = range(1, N_EPOCHS + 2)
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

    plt.plot(x, test_auc_list, 'b:o', label="test_auc")
    plt.plot(x, train_auc_list, 'r:o', label="train_auc")
    plt.plot(x, test_aupr_list, 'b-o', label="test_aupr")
    plt.plot(x, train_aupr_list, 'r-o', label="train_aupr")
    plt.ylabel("auc & aupr")
    plt.xlabel("epoch")
    plt.show()
