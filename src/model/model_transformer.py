import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

embedding_matrix = torch.as_tensor(np.load("embedding_matrix.npy"))
MAX_LEN_en = 3000  # seq_lens
MAX_LEN_pr = 2000  # seq_lens
NB_WORDS = 4097  # one-hot dim
EMBEDDING_DIM = 100