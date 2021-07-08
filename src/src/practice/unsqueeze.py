import torch

a = torch.tensor([[1, 1, 1], [2, 2, 2]])


b = a.unsqueeze(2)

print(a)
print(b)
