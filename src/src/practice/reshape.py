import torch
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import itertools


y_list = []
y = torch.as_tensor([1, 1, 1, 1]).tolist()
y_list.append(y)
y = torch.as_tensor([1, 1, 1, 1]).tolist()
y_list.append(y)
y = torch.as_tensor([1, 0]).tolist()
y_list.append(y)

print(y_list)

y = list(itertools.chain.from_iterable(y_list))
print(y)

y_pred = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0]]
y_pred = list(itertools.chain.from_iterable(y_pred))


score = roc_auc_score(y, y_pred)
print(score)
score = average_precision_score(y, y_pred)
print(score)