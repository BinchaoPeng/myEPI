import torch

print(torch.cuda.is_available())

device = torch.device('cuda:0')

print(device.type)
print(device.index)
print(device)
