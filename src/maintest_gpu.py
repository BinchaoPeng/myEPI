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
import model.model_longformer_gru as model_longformer_gru
import model.model_longformer_lstm as model_longformer_lstm
import model.model_elmo_1 as model_elmo_1
import model.model_elmo_2 as model_elmo_2

device, USE_GPU = use_gpu_first()
print("device:", device)

"""
EarlyStopping parameter
"""
# 初始化 early_stopping 对象
patience = 5  # 当验证集损失在连续patience次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合

"""
Hyper parameter
"""
N_EPOCHS = 30
batch_size = 12
# 加载数据（batch）的线程数目
# if time of loading data is more than the time of training,we add num_works to reduce loading time
num_works = 16
lr = 0.00001
"""
cell and feature choose
"""
names = ['PBC', 'pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[5]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[3]

np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽

trainSet = EPIDataset(cell_name, feature_name=feature_name)
len_trainSet = len(trainSet)
print("trainSet data len:", len(trainSet))
trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=num_works)
len_trainLoader = len(trainLoader)
print("trainLoader len:", len_trainLoader)

testSet = EPIDataset(cell_name, feature_name=feature_name, is_train_set=False)
len_testSet = len(testSet)
testLoader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=num_works, drop_last=True)

"""
base
"""
# module = modelBase.EPINet()
module = model_elmo_1.EPINet()
# module = model_elmo_2.EPINet()
# module = model_longformer_lstm.EPINet()
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
if hasattr(module, 'longformer'):
    for para in module.longformer.parameters():
        para.requires_grad = False
    # print(module.parameters())

"""
loss and optimizer
"""
# criterion = nn.BCELoss(reduction='sum')
criterion = nn.BCEWithLogitsLoss(reduction='sum')
# optimizer = optim.Adam(module.parameters(), lr=lr)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, module.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)
"""
total info
"""
print("==" * 20)
print("==USE_GPU:", USE_GPU)
print("==DEVICE:", device)
print("==")
print("==CELL_NAME:", cell_name)
print("==FEATURE_NAME:", feature_name)
print("==MODEL_NAME:", model_name)
print("==")
print("==N_EPOCHS:", N_EPOCHS)
print("==num_workers:", num_works)
print("==batch_size:", batch_size)
print("==lr:", lr)
print("==" * 20)

if __name__ == '__main__':

    start_time = time.time()
    loss_list = []
    test_auc_list = []
    test_aupr_list = []

    # train_auc_list = []
    # train_aupr_list = []

    # init earlyStopping
    best_acc = 0.5
    es_count = 0
    for epoch in range(1, N_EPOCHS + 1):

        # train
        module.train()
        loss = trainModel(len_trainLoader // 20, start_time, len_trainSet, epoch, trainLoader,
                               module, criterion, optimizer, scheduler=None)
        scheduler.step()
        # train_auc_list.append(auc)
        # train_aupr_list.append(aupr)
        loss_list.append(loss)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        print(f"==================[{time_since(start_time)}]train: EPOCH {epoch} is over!==================")
        # test
        module.eval()
        auc, aupr = testModel(testLoader, module)
        test_auc_list.append(auc)
        test_aupr_list.append(aupr)
        print(f"==================[{time_since(start_time)}]test: EPOCH {epoch} is over!==================")

        # print("============================test: ACC is", acc, "=======================================")
        # torch.save(module, r'..\model\%sModule-%s.pkl' % (name, str(epoch)))
        torch.save(module, r'../model/model-%s-%s.pkl' % (cell_name, str(epoch)))  # must use /
        print("==================saved model !", "==================")
        # early_stopping
        val_acc = auc
        if val_acc > best_acc:
            best_acc = val_acc
            es_count = 0
        else:
            es_count += 1
            print("===\nEarly stopping counter {} of {}!!\n===".format(es_count, patience))
            if es_count > (patience - 1):
                print("\n===Early stopping!!===\n")
                break
    print("\n\n[CELL_NAME:", cell_name, "FEATURE_NAME:", feature_name, "MODEL_NAME:", model_name, "]")
    # polt
    drawMetrics(loss,test_auc_list, test_aupr_list, cell_name, feature_name, model_name)
