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

        self.enhancer_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.enhancer_embedding.weight.shape)
        self.promoter_embedding = nn.Embedding(NB_WORDS, EMBEDDING_DIM, _weight=embedding_matrix)
        # print("net:", self.promoter_embedding.weight.shape)

        # encoder_layer = nn.TransformerEncoderLayer(nhead=10, d_model=EMBEDDING_DIM)
        # self.enhancer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)
        # self.promoter_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

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
                          hidden_size=256,
                          num_layers=num_layers,
                          bidirectional=bidirectional)
        self.gru_1 = nn.GRU(input_size=512,
                            hidden_size=50,
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        # self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
        #                     output_size)

        encoder_layer = nn.TransformerEncoderLayer(nhead=8, d_model=512)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

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
        # x_en = self.enhancer_encoder(x_en.type(torch.float32))
        # # print("encoder(x_en):", x_en.shape)
        # x_pr = self.promoter_encoder(x_pr.type(torch.float32))
        # # print("encoder(x_pr):", x_pr.shape)

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
        hidden = self._init_hidden(batch_size, 256)

        # sql_length = torch.LongTensor([batch_size for i in range(0, batch_size)])
        # gru_input = pack_padded_sequence(X, sql_length, batch_first=True)

        # output: [h1,h2,h3,...,hn]  (seqSize,batch,hidden_size)   seqSize: dim of input
        # hidden: hN (numLayers,batch,hidden_size)
        output, hidden = self.gru(X,  # seq (seqSize,batch,input_size)
                                  )  # h0 (numLayers,batch,hidden_size)

        # print("gru_output:", output.shape)
        # print("gru_hidden:", hidden.shape)

        # if self.n_directions == 2:
        #     hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # else:
        #     hidden_cat = hidden[-1]

        # print("gru:", hidden_cat.shape)

        X = self.encoder(output)
        # print("encoder(output):", X.shape)

        output, hidden = self.gru_1(X,  # seq (seqSize,batch,input_size)
                                    )  # h0 (numLayers,batch,hidden_size)

        # print("gru1_output:", output.shape)
        # print("gru1_hidden:", hidden.shape)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        print("gru:", hidden_cat.shape)

        fc_output = self.fc(hidden_cat)
        print("fc:", fc_output.shape)

        fc_output = torch.sigmoid(fc_output)
        print("fc_sigmoid:", fc_output.shape)
        # print("fc:", fc_output)

        fc_output = fc_output.view(-1)
        print("view:", fc_output.shape)
        print("view:", fc_output)

        return fc_output

    def _init_hidden(self, batch_size, hidden_size):
        return torch.zeros(self.num_layers * self.n_directions,
                           batch_size,
                           hidden_size)
