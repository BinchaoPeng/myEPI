"""
use unsqueeze to add a dim and delete embedding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_matrix = torch.as_tensor(np.load("embedding_matrix.npy"))
MAX_LEN_en = 3000  # seq_lens
MAX_LEN_pr = 2000  # seq_lens
NB_WORDS = 4097  # one-hot dim
EMBEDDING_DIM = 100


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.enhancer_conv_layer = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.enhancer_conv_layer.weight.shape)
        self.enhancer_max_pool_layer = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.promoter_conv_layer = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.promoter_conv_layer.weight.shape)
        self.promoter_max_pool_layer = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.bn = nn.BatchNorm1d(num_features=64)
        self.dt = nn.Dropout(p=0.5)

        self.gru = nn.GRU(input_size=64,
                          hidden_size=50,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        # self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
        #                     output_size)

        self.fc = nn.Linear(100, 1)

    def forward(self, x):

        x_en = torch.as_tensor(x[0].unsqueeze(2), dtype=torch.float)
        # print("x_en:", x_en.shape)  #(Batch_size,3000)
        # print("x_en:", x_en)
        """
        x_en: torch.Size([B, 712])
        """

        x_pr = torch.as_tensor(x[1].unsqueeze(2), dtype=torch.float)
        # print("x_pr:", x_pr.shape) #(Batch_size,2000)
        # print("x_pr:", x_pr)
        """
        x_pr: torch.Size([B, 712])
        """
        # x_en = x_en.t()
        # x_pr = x_pr.t()

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

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        # print("gru:", hidden_cat.shape)
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
