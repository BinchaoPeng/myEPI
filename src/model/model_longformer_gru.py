import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig
from utils.utils_dl import use_gpu_first

EMBEDDING_DIM = 768
max_value = 2708
device, USE_GPU = use_gpu_first()


class EPINet(nn.Module):
    def __init__(self, num_layers=1, bidirectional=True):
        super(EPINet, self).__init__()

        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        model_name = 'pre-model/' + 'longformer-base-4096'
        config = LongformerConfig.from_pretrained(model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.longformer = LongformerModel.from_pretrained(model_name, config=config)  # (B,2653,768)
        # for param in self.longformer.base_model.parameters():
        #     param.requires_grad = False

        self.conv1 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=64, kernel_size=40)
        # print("net:", self.enhancer_conv_layer.weight.shape)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=20, stride=20)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=20)
        # print("net:", self.promoter_conv_layer.weight.shape)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=10, stride=3)
        # print("net:", self.enhancer_conv_layer.weight.shape)

        self.bn = nn.BatchNorm1d(num_features=32)  # input_size / embedding_dim
        self.dt = nn.Dropout(p=0.5)

        self.gru = nn.GRU(input_size=32,
                          hidden_size=50,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        # self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
        #                     output_size)

        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        """
        x: en+</s>+pr
        """

        encoded_inputs = self.tokenizer(x, padding='max_length', max_length=max_value, return_tensors='pt')
        encoded_inputs.to(device)
        X_enpr_tensor = self.longformer(**encoded_inputs)[0]
        # print("X_enpr_tensor:", X_enpr_tensor.shape)  # (Batch_size,2651,768) (B,S,I)
        # print("X_enpr_tensor:", X_enpr_tensor)

        X_enpr_tensor = X_enpr_tensor.permute(0, 2, 1)
        # print("X_enpr_tensor,permute(0,2,1):", X_enpr_tensor.shape)  # (B,768,2651)

        x_enpr = self.conv1(X_enpr_tensor)
        # print("conv1(X_enpr_tensor):", x_enpr.shape)
        x_enpr = F.relu(x_enpr)
        # print("conv1_relu(x_enpr):", x_enpr.shape)
        x_enpr = self.max_pool_1(x_enpr)
        # print("max_pool_1(x_enpr):", x_enpr.shape)

        x_enpr = self.conv2(x_enpr)
        # print("conv2(x_enpr):", x_enpr.shape)
        x_enpr = F.relu(x_enpr)
        # print("conv2_relu(x_enpr):", x_enpr.shape)
        x_enpr = self.max_pool_2(x_enpr)
        # print("max_pool_2(x_enpr):", x_enpr.shape)

        x_enpr = self.bn(x_enpr)
        # print("bn:", x_enpr.shape)

        x_enpr = self.dt(x_enpr)
        # print("dt:", x_enpr.shape)

        x_enpr = x_enpr.permute(2, 0, 1)
        # print("dt:", x_enpr.shape)

        batch_size = x_enpr.size(1)
        # hidden = self._init_hidden(batch_size, 50)
        # hidden.to("cuda:0")
        # sql_length = torch.LongTensor([batch_size for i in range(0, batch_size)])
        # gru_input = pack_padded_sequence(x_enpr, sql_length, batch_first=True)

        # output: [h1,h2,h3,...,hn]  (seqSize,batch,hidden_size)   seqSize: dim of input
        # hidden: hN (numLayers,batch,hidden_size)
        # output, hidden = self.gru(x_enpr,  # seq (seqSize,batch,input_size)
        #                           hidden)  # h0 (numLayers,batch,hidden_size)
        output, hidden = self.gru(x_enpr)  # seq (seqSize,batch,input_size)

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
