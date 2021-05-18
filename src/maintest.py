import torch
import torch.nn as nn
import torch.optim as optim
import time, sys
import numpy as np
from testModel import testModel
from trainModel import trainModel, time_since
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from draw_metrics import drawMetrics

from EPIDataset import EPIDataset
import model.modelBase as modelBase
import model.model_gru3 as model_gru3
import model.model_transformer as model_transformer
import model.model_transformer_1 as model_transformer_1
import model.model_transformer_2 as model_transformer_2
import model.model_pseknc_1 as model_pseknc_1
import model.model_pseknc_2 as model_pseknc_2
import model.model_pseknc_3 as model_pseknc_3
import model.model_longformer_1 as model_longformer_1


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Hyper parameter
"""
USE_GPU = True if get_device() in 'cuda' else False
N_EPOCHS = 40
batch_size = 64
num_works = 0
lr = 0.00001
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[5]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug']
feature_name = feature_names[2]

np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽

trainSet = EPIDataset(cell_name, feature_name=feature_name)
len_trainSet = len(trainSet)
print("trainSet data len:", len(trainSet))
trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=num_works)
print("trainLoader len:", len(trainLoader))

testSet = EPIDataset(cell_name, feature_name=feature_name, is_train_set=False)
len_testSet = len(testSet)
testLoader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=num_works, drop_last=True)

"""
base
"""
# module = modelBase.EPINet()

module = model_longformer_1.EPINet()
# module = model_pseknc_2.EPINet()
# module = model_pseknc_1.EPINet()
# module = model_transformer_2.EPINet()
# module = model_transformer_1.EPINet()
# module = model_transformer.EPINet()
# module = model_gru3.EPINet()
# module = model_transformer.EPINet()
model_name = module.__class__.__module__

# 这里是一般情况，共享层往往不止一层，所以做一个for循环
for para in module.longformer.parameters():
    para.requires_grad = False
# print(module.parameters())
"""
loss and optimizer
"""
criterion = nn.BCELoss(reduction='sum')
# optimal = optim.Adam(module.parameters(), lr=lr)
optimal = optim.Adam(filter(lambda p: p.requires_grad, module.parameters()), lr=lr)

if __name__ == '__main__':

    print("===CELL_NAME:", cell_name, "FEATURE_NAME:", feature_name, "MODEL_NAME:", model_name, "===\n\n")

    if USE_GPU:
        device = torch.device("cuda:0")
        module.to(device)

    start = time.time()
    acc_list = []
    test_auc_list = []
    test_aupr_list = []

    train_auc_list = []
    train_aupr_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # train
        module.train()
        auc, aupr = trainModel(20, start, len_trainSet, epoch, trainLoader, module, criterion, optimal)
        train_auc_list.append(auc)
        train_aupr_list.append(aupr)
        print(f"============================[{time_since(start)}]train: EPOCH {epoch} is over!================")

        # test
        module.eval()
        auc, aupr = testModel(testLoader, module)
        test_auc_list.append(auc)
        test_aupr_list.append(aupr)
        print(f"============================[{time_since(start)}]test: EPOCH {epoch} is over!================")

        # print("============================test: ACC is", acc, "=======================================")
        # torch.save(module, r'..\model\%sModule-%s.pkl' % (name, str(epoch)))
        torch.save(module, r'../model/model-%s-%s.pkl' % (cell_name, str(epoch)))  # must use /
        print("============================saved model !", "=======================================")

    print("\n\n[CELL_NAME:", cell_name, "FEATURE_NAME:", feature_name, "MODEL_NAME:", model_name, "]")
    # polt
    drawMetrics(N_EPOCHS, train_auc_list, test_auc_list, train_aupr_list, test_aupr_list,
                cell_name, feature_name, model_name)
