import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids

from utils.utils_dl import use_gpu_first

device, USE_GPU = use_gpu_first()


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        # [batch_size, seq_len, embedding_dim=256]
        # options_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        # weight_file = "pre-model/elmo_model/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        # EMBEDDING_DIM = 256

        # [batch_size, seq_len, embedding_dim=512]
        # options_file = "pre-model/elmo_model/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        # weight_file = "pre-model/elmo_model/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
        # EMBEDDING_DIM = 512

        # [batch_size, seq_len, embedding_dim=1024]
        options_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "pre-model/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        EMBEDDING_DIM = 1024

        self.elmo_en = Elmo(options_file, weight_file, num_output_representations=1, requires_grad=False, dropout=0.5)
        self.elmo_pr = Elmo(options_file, weight_file, num_output_representations=1, requires_grad=False, dropout=0.5)

        self.conv1_en = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.enhancer_conv_layer.weight.shape)
        self.max_pool_1_en = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.conv2_en = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=20)
        # print("net:", self.promoter_conv_layer.weight.shape)
        self.max_pool_2_en = nn.MaxPool1d(kernel_size=10, stride=3)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.bn_en = nn.BatchNorm1d(num_features=32)  # input_size / embedding_dim
        self.dt_en = nn.Dropout(p=0.5)

        self.conv1_pr = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.enhancer_conv_layer.weight.shape)
        self.max_pool_1_pr = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.conv2_pr = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=20)
        # print("net:", self.promoter_conv_layer.weight.shape)
        self.max_pool_2_pr = nn.MaxPool1d(kernel_size=10, stride=3)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.bn_pr = nn.BatchNorm1d(num_features=32)  # input_size / embedding_dim
        self.dt_pr = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=50,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

        # self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
        #                     output_size)

        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        """
        x: en pr
        """
        en = [item.split(" ")[0] for item in x]
        pr = [item.split(" ")[1] for item in x]
        en_ids = batch_to_ids(en)
        pr_ids = batch_to_ids(pr)
        # print("en_ids type:", en_ids.shape)
        en_ids = en_ids.to(device)
        pr_ids = pr_ids.to(device)

        X_en_tensor = self.elmo_en(en_ids)['elmo_representations'][0]
        X_pr_tensor = self.elmo_pr(pr_ids)['elmo_representations'][0]

        # print("X_en_tensor:", X_en_tensor.shape)  # (B,S,I)
        # print("X_enpr_tensor:", X_enpr_tensor)

        X_en_tensor = X_en_tensor.permute(0, 2, 1)
        X_pr_tensor = X_pr_tensor.permute(0, 2, 1)
        # print("X_enpr_tensor,permute(0,2,1):", X_enpr_tensor.shape)  # (B,768,2651)

        x_en = self.conv1_en(X_en_tensor)
        x_pr = self.conv1_pr(X_pr_tensor)
        # print("conv1(x_en):", x_en.shape)
        x_en = F.relu(x_en)
        x_pr = F.relu(x_pr)
        # print("conv1_relu(x_en):", x_en.shape)
        x_en = self.max_pool_1_en(x_en)
        x_pr = self.max_pool_1_pr(x_pr)
        # print("max_pool_1(x_en):", x_en.shape)

        x_en = self.conv2_en(x_en)
        x_pr = self.conv2_pr(x_pr)
        # print("conv2(x_en):", x_en.shape)
        x_en = F.relu(x_en)
        x_pr = F.relu(x_pr)
        # print("conv2_relu(x_en):", x_en.shape)
        x_en = self.max_pool_2_en(x_en)
        x_pr = self.max_pool_2_pr(x_pr)
        # print("max_pool_2(x_en):", x_en.shape)

        x_en = self.bn_en(x_en)
        x_pr = self.bn_pr(x_pr)
        # print("bn:", x_en.shape)

        x_en = self.dt_en(x_en)
        x_pr = self.dt_pr(x_pr)
        # print("dt:", x_en.shape)

        x_enpr = torch.cat([x_en, x_pr], 2)

        x_enpr = x_enpr.permute(2, 0, 1)
        # print("dt:", x_en.shape)

        output, (hidden, c_n) = self.lstm(x_enpr)  # seq (seqSize,batch,input_size)

        # print("gru_output:", output.shape)
        # print("gru_hidden:", hidden.shape)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        # print("gru:", hidden_cat.shape)
        fc_output = self.fc(hidden_cat)
        # print("fc:", fc_output.shape)

        # fc_output = torch.sigmoid(fc_output)
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
