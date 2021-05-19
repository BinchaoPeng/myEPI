import torch
import numpy as np

data = ('A', 'd', 'v')

data = np.array(data)
# data = torch.ones((3, 5))
data = torch.from_numpy(data)
print(data)

# data = torch.from_numpy(data)
data.to("cuda")
