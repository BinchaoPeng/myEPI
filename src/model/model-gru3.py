import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

embedding_matrix = torch.as_tensor(np.load("../embedding_matrix.npy"))
MAX_LEN_en = 3000  # seq_lens
MAX_LEN_pr = 2000  # seq_lens
NB_WORDS = 4097  # one-hot dim
EMBEDDING_DIM = 100


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.enhancer_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.enhancer_embedding.weight.shape)
        self.promoter_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.promoter_embedding.weight.shape)

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

        self.gru_1 = nn.GRU(input_size=64, hidden_size=125, bidirectional=bidirectional,
                            num_layers=self.num_layers, dropout=0.5,
                            batch_first=True)

        self.gru_2 = nn.GRU(input_size=246, hidden_size=128,
                            num_layers=self.num_layers, dropout=0.5,
                            batch_first=True)

        self.gru_3 = nn.GRU(input_size=16, hidden_size=128,
                            num_layers=self.num_layers, dropout=0.5,
                            batch_first=True)

        self.fc_1 = nn.Sequential(nn.Linear(128, 16),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.BatchNorm1d(16, momentum=0.5),
                                  nn.Dropout(0.25),

                                  )
        self.fc_2 = nn.Linear(16, 1)

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

        X = torch.cat([x_en, x_pr], 2)
        # print("cat:", X.shape)

        X = self.bn(X)
        # print("bn:", X.shape)

        """
        dt: torch.Size([8, 64, 246])
        permute(0,2,1): torch.Size([8, 246, 64])
        gru_1 shape: torch.Size([8, 246, 250])
        contiguous shape: torch.Size([8, 246, 250])
        permute shape: torch.Size([8, 250, 246])
        gru_2 shape: torch.Size([8, 250, 128])
        [:, -1, :] shape: torch.Size([8, 128])
        contiguous shape: torch.Size([8, 128])
        view shape: torch.Size([8, 8, 16])
        gru_3 shape: torch.Size([8, 8, 128])
        [:, -1, :] shape: torch.Size([8, 128])
        contiguous shape: torch.Size([8, 128])
        view shape: torch.Size([8, 128])
        X shape: torch.Size([8, 16])
        fc: torch.Size([8, 1])
        fc_sigmoid: torch.Size([8, 1])
        """

        X = self.dt(X)  # (B,64,246)  (B,I,S)
        # print("dt:", X.shape)

        X = X.permute(0, 2, 1)
        # print("permute(0,2,1):", X.shape)   # # (B,246,64)  (B,S,I)

        X, _ = self.gru_1(X)
        # print("gru_1 shape:", X.shape)  # X shape: torch.Size([1, 101, 250])
        X = X.contiguous()
        # print("contiguous shape:", X.shape)  # X shape: torch.Size([1, 101, 250])
        X = X.permute(0, 2, 1)
        # print("permute shape:", X.shape)  # X shape: torch.Size([1, 250, 101])
        X, _ = self.gru_2(X)
        # print("gru_2 shape:", X.shape)  # X shape: torch.Size([1, 250, 128])
        X = X[:, -1, :]
        # print("[:, -1, :] shape:", X.shape)  # X shape: torch.Size([1, 128])
        X = X.contiguous()
        # print("contiguous shape:", X.shape)  # X shape: torch.Size([1, 128])
        X = X.view(X.shape[0], 8, 16)
        # print("view shape:", X.shape)  # X shape: torch.Size([1, 8, 16])

        X, _ = self.gru_3(X)
        # print("gru_3 shape:", X.shape)  # X shape: torch.Size([1, 8, 128])
        X = X[:, -1, :]  # X shape: torch.Size([1, 128])
        # print("[:, -1, :] shape:", X.shape)
        X = X.contiguous()
        # print("contiguous shape:", X.shape)  # X shape: torch.Size([1, 128])
        X = X.view(X.shape[0], 128)
        # print("view shape:", X.shape)  # X shape: torch.Size([1, 128])

        X = self.fc_1(X)
        # print("X shape:", X.shape)

        fc_output = self.fc_2(X)
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