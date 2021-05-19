import torch

x = torch.ones((3, 5))
print(x)

print('type(x): ', type(x))
print('x.dtype: ', x.dtype)
print('x.device:', x.device)

x = x.to("cuda")
print('type(x): ', type(x))
print('x.dtype: ', x.dtype)
print('x.device:', x.device)
