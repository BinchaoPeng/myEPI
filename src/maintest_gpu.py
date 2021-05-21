import torch
import torch.nn as nn
import torch.optim as optim
import time, sys
import numpy as np
from testModel import testModel
from trainModel import trainModel
from utils import time_since, use_gpu_first
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
# import model.model_longformer_gru as model_longformer_gru
import model.model_longformer_lstm as model_longformer_lstm


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device, USE_GPU = use_gpu_first()
print("device:", device)

"""
Hyper parameter
"""
N_EPOCHS = 40
batch_size = 16
num_works = 0
lr = 0.00001
"""
cell and feature choose
"""
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
module = model_longformer_lstm.EPINet()
# module = model_longformer_gru.EPINet()
# module = model_pseknc_2.EPINet()
# module = model_pseknc_1.EPINet()
# module = model_transformer_2.EPINet()
# module = model_transformer_1.EPINet()
# module = model_transformer.EPINet()
# module = model_gru3.EPINet()
# module = model_transformer.EPINet()
model_name = module.__class__.__module__
module.to(device)

# 这里是一般情况，共享层往往不止一层，所以做一个for循环
for para in module.longformer.parameters():
    para.requires_grad = False
# print(module.parameters())
"""
loss and optimizer
"""
criterion = nn.BCELoss(reduction='sum')
# optimizer = optim.Adam(module.parameters(), lr=lr)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, module.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)
"""
total info
"""
print("==" * 20)
print("==USE_GPU:", USE_GPU)
print("==DEVICE:", device)

print("==CELL_NAME:", cell_name)
print("==FEATURE_NAME:", feature_name)
print("==MODEL_NAME:", model_name)

print("==N_EPOCHS:", N_EPOCHS)
print("==batch_size:", batch_size)
print("==lr:", lr, )
print("==" * 20)

if __name__ == '__main__':

    start_time = time.time()
    acc_list = []
    test_auc_list = []
    test_aupr_list = []

    train_auc_list = []
    train_aupr_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # train
        module.train()
        auc, aupr = trainModel(120, start_time, len_trainSet, epoch, trainLoader,
                               module, criterion, optimizer, scheduler=None)
        scheduler.step()
        train_auc_list.append(auc)
        train_aupr_list.append(aupr)
        print(f"============================[{time_since(start_time)}]train: EPOCH {epoch} is over!================")
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # test
        module.eval()
        auc, aupr = testModel(testLoader, module)
        test_auc_list.append(auc)
        test_aupr_list.append(aupr)
        print(f"============================[{time_since(start_time)}]test: EPOCH {epoch} is over!================")

        # print("============================test: ACC is", acc, "=======================================")
        # torch.save(module, r'..\model\%sModule-%s.pkl' % (name, str(epoch)))
        torch.save(module, r'../model/model-%s-%s.pkl' % (cell_name, str(epoch)))  # must use /
        print("============================saved model !", "=======================================")

    print("\n\n[CELL_NAME:", cell_name, "FEATURE_NAME:", feature_name, "MODEL_NAME:", model_name, "]")
    # polt
    drawMetrics(N_EPOCHS, train_auc_list, test_auc_list, train_aupr_list, test_aupr_list,
                cell_name, feature_name, model_name)
