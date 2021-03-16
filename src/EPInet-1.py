import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

"""
Hyper parameter
"""
N_EPOCHS = 2
batch_size = 8
num_works = 8
lr = 0.001

embedding_matrix = torch.as_tensor(np.load("embedding_matrix.npy"))
MAX_LEN_en = 3000  # seq_lens
MAX_LEN_pr = 2000  # seq_lens
NB_WORDS = 4097  # one-hot dim
EMBEDDING_DIM = 100

"""
load dataset
"""
names = ['PBC', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
name = names[0]
trainPath = r'../data/epivan/%s/%s_train.npz' % (name, name)
testPath = r'../data/epivan/%s/%s_test.npz' % (name, name)


class EPIDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = trainPath if is_train_set else testPath
        data = np.load(filename)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
        self.X_en = data[data.files[0]]
        self.X_pr = data[data.files[1]]
        self.X = [item for item in zip(self.X_en, self.X_pr)]
        # print(type(self.X))
        self.y = data[data.files[2]]
        self.len = len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽

trainSet = EPIDataset()
print("trainSet data len:", len(trainSet))
trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=num_works)
print("trainLoader len:", len(trainLoader))

testSet = EPIDataset(is_train_set=False)
testLoader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=num_works, drop_last=True)

"""
construct module
"""


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        # enhancers = Input(shape=(MAX_LEN_en,))
        # promoters = Input(shape=(MAX_LEN_pr,))
        # emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(enhancers)
        # emb_pr = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(promoters)

        self.enhancer_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.enhancer_embedding.weight.shape)
        self.promoter_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.promoter_embedding.weight.shape)
        # enhancer_conv_layer = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(emb_en)
        # enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer)

        self.enhancer_conv_layer = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.enhancer_conv_layer.weight.shape)
        self.enhancer_max_pool_layer = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        # promoter_conv_layer = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(emb_pr)
        # promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer)

        self.promoter_conv_layer = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.promoter_conv_layer.weight.shape)
        self.promoter_max_pool_layer = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        # merge_layer = Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
        # bn = BatchNormalization()(merge_layer)
        # dt = Dropout(0.5)(bn)

        self.bn = nn.BatchNorm1d(num_features=64)
        self.dt = nn.Dropout(p=0.5)

        # l_gru = Bidirectional(GRU(50, return_sequences=True))(dt)
        # l_att = AttLayer(50)(l_gru)

        # self.gru = nn.GRU(64,  # input
        #                   50,  # output
        #                   1,
        #                   bidirectional=True)

        self.gru = nn.GRU(input_size=64,
                          hidden_size=50,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        # self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
        #                     output_size)

        self.transformer = nn.Transformer()

        self.fc = nn.Linear(100, 1)

        # preds = Dense(1, activation='sigmoid')(l_att)

    def forward(self, x):

        x_en = torch.as_tensor(x[0], dtype=torch.long)
        # print("x_en:", x_en.shape)  #(Batch_size,3000)
        # print("x_en:", x_en)
        """
        x_en: torch.Size([64, 3000])
        x_en: tensor([  [   0,    0,    0,  ...,  306, 1224,  798],
                        [   0,    0,    0,  ...,  612, 2448, 1598],
                        [   0,    0,    0,  ...,  635, 2539, 1961],
                         ...,
                        [   0,    0,    0,  ..., 1983, 3833, 3043],
                        [   0,    0,    0,  ..., 2080,  126,  503],
                        [   0,    0,    0,  ...,    1,    4,   15]])
        """

        x_pr = torch.as_tensor(x[1], dtype=torch.long)
        # print("x_pr:", x_pr.shape) #(Batch_size,2000)
        # print("x_pr:", x_pr)
        """
        x_pr: torch.Size([64, 2000])
        x_pr: tensor([[   0,    0,    0,  ..., 2827, 3113,  161],
                [   0,    0,    0,  ..., 3967, 3579, 2027],
                [   0,    0,    0,  ..., 1386, 1446, 1687],
                ...,
                [   0,    0,    0,  ...,  291, 1161,  546],
                [   0,    0,    0,  ..., 2985, 3747, 2699],
                [   0,    0,    0,  ..., 2654, 2421, 1491]])
        """
        # x_en = x_en.t()
        # x_pr = x_pr.t()

        x_en = self.enhancer_embedding(x_en)
        # print("embedding(x_en):", x_en.shape)
        x_pr = self.promoter_embedding(x_pr)
        # print("embedding(x_pr):", x_pr.shape)
        """
        data: torch.Size([64, 3000, 100])
        data: torch.Size([64, 2000, 100])
        """

        x_en = x_en.permute(0, 2, 1)
        # print("x_en.permute:", x_en.shape)
        x_pr = x_pr.permute(0, 2, 1)
        # print("x_en.permute:", x_pr.shape)

        x_en = x_en.type(torch.float32)
        # print(x_en.dtype)
        x_en = self.enhancer_conv_layer(x_en)
        # print("enhancer_conv_layer(x_en):", x_en.shape)
        x_pr = self.promoter_conv_layer(x_pr.type(torch.float32))
        # print("promoter_conv_layer(x_pr):", x_pr.shape)

        x_en = F.relu(x_en)
        # print("relu(x_en):", x_en.shape)
        x_pr = F.relu(x_pr)
        # print("relu(x_pr):", x_pr.shape)

        x_en = self.enhancer_max_pool_layer(x_en)
        # print("max_pool_layer(x_en):", x_en.shape)
        x_pr = self.promoter_max_pool_layer(x_pr)
        # print("max_pool_layer(x_pr):", x_pr.shape)

        # x_en = x_en.permute(0, 2, 1)
        # print("data:", x_en.shape)
        # x_pr = x_pr.permute(0, 2, 1)
        # print("data:", x_pr.shape)

        X = torch.cat([x_en, x_pr], 2)
        # print("cat:", X.shape)

        X = self.bn(X)
        # print("bn:", X.shape)

        X = self.dt(X)
        # print("dt:", X.shape)

        X = X.permute(2, 0, 1)
        # print("dt:", X.shape)
        batch_size = X.size(1)
        hidden = self._init_hidden(batch_size, 50)

        # sql_length = torch.LongTensor([batch_size for i in range(0, batch_size)])
        # gru_input = pack_padded_sequence(X, sql_length, batch_first=True)

        # output: [h1,h2,h3,...,hn]  (seqSize,batch,hidden_size)   seqSize: dim of input
        # hidden: hN (numLayers,batch,hidden_size)
        output, hidden = self.gru(X,  # seq (seqSize,batch,input_size)
                                  hidden)  # h0 (numLayers,batch,hidden_size)

        # print("gru_output:", output.shape)
        # print("gru_hidden:", hidden.shape)

        # if self.n_directions == 2:
        #     hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # else:
        #     hidden_cat = hidden[-1]
        # print("gru:", hidden_cat.shape)

        hidden_cat = self.transformer(output)
        print("transformer:", hidden_cat.shape)

        fc_output = self.fc(hidden_cat)
        # print("fc:", fc_output.shape)

        fc_output = torch.sigmoid(fc_output)
        # print("fc_sigmoid:", fc_output.shape)
        # print("fc:", fc_output)

        fc_output = fc_output.view(-1)
        # print("view:", fc_output.shape)
        # print("view:", fc_output)

        return fc_output

    def _init_hidden(self, batch_size, hidden_size):
        return torch.zeros(self.num_layers * self.n_directions,
                           batch_size,
                           hidden_size)


module = EPINet()
# print(module.parameters())
"""
loss and optimizer
"""
criterion = nn.BCELoss(reduction='sum')
optimal = optim.Adam(module.parameters(), lr=lr)

"""
train and set
"""


def time_since(start):
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(epoch):
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
        for item_pred, item_test in zip(pred, y):
            y_pred.append(item_pred)
            y_test.append(item_test)

        # print(y_pred, y)
        loss = criterion(pred.type(torch.float), y.type(torch.float))

        optimal.zero_grad()
        loss.backward()

        optimal.step()

        total_loss += loss.item()
        if i % 5 == 0:
            # print("len(trainLoader):", len(trainLoader))
            # print("len(x)", len(x))
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(x[0])}/{len(trainSet)}]', end='')
            print(f'loss = {total_loss / (i * len(x[0]))}')

    y_test = np.array(y_test).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    auc = roc_auc_score(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)

    print("train AUC : ", auc)
    print("train AUPR : ", aupr)
    return auc, aupr


def evaluate_accuracy(x, y, net):
    out = net(x)
    correct = (out.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct / n


def testModel():
    correct = 0
    total = len(testSet)
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
        auc, aupr = trainModel(epoch)
        train_auc_list.append(auc)
        train_aupr_list.append(aupr)
        print(f"============================[{time_since(start)}]train: EPOCH {epoch} is over!================")
        auc, aupr = testModel()
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