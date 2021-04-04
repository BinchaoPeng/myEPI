import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_matrix = torch.as_tensor(np.load("embedding_matrix.npy"))
MAX_LEN_en = 286  # seq_lens
MAX_LEN_pr = 286  # seq_lens
NB_WORDS = 4097  # one-hot dim
EMBEDDING_DIM = 100


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=2,
                          hidden_size=50,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True)

        self.fc = nn.Linear(100, 1)

    def forward(self, x):

        x_en = torch.as_tensor(x[0], dtype=torch.float)
        # print("x_en:", x_en.shape)  #(Batch_size,3000)
        # print("x_en:", x_en)

        x_pr = torch.as_tensor(x[1], dtype=torch.float)
        # print("x_pr:", x_pr.shape) #(Batch_size,2000)
        # print("x_pr:", x_pr)

        X = torch.cat([x_en, x_pr], 1)
        # print("cat:", X.shape)
        batch_size = X.size(0)
        X = X.reshape((batch_size,286,2))
        # print("reshape:", X.shape)

        # batch_size = X.size(1)
        # hidden = self._init_hidden(batch_size, 1)

        # sql_length = torch.LongTensor([batch_size for i in range(0, batch_size)])
        # gru_input = pack_padded_sequence(X, sql_length, batch_first=True)

        # output: [h1,h2,h3,...,hn]  (seqSize,batch,hidden_size)   seqSize: dim of input
        # hidden: hN (numLayers,batch,hidden_size)
        output, hidden = self.gru(X,  # seq (seqSize,batch,input_size)
                                  )  # h0 (numLayers,batch,hidden_size)

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
