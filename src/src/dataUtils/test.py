import numpy as np

en_data = np.array([1, 2, 3])
pr_data = np.array([1, 2, 3])

print(en_data)

x_data = [np.array(item)
          for item in zip(en_data, pr_data)]
print(x_data[0].shape)

# In[ ]:npz 数据信息
import numpy as np

data = np.load("data/epivan/IMR90/im_IMR90_train.npz")
np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽
print(data.files)
print(type(data))
print(data[data.files[0]])
print(type(data[data.files[0]]))
print(data[data.files[0]].shape)

# In[ ]:
import torch
import torch.nn as nn

m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
print(input.shape)
ran = torch.randn(2, 3, 4)
print(ran.shape)
print(ran)
output = m(input)
print(output.shape)

# In[ ]:
import numpy as np

"""
write 3-dim np_arr to csv
"""
l = [[[1, 1, 1],
      [1, 1, 1]],
     [[2, 2, 2],
      [2, 2, 2]]]

arr = np.array(l)
with open(r"src/dataUtils/dtest.csv", 'ab') as f:

    for item in arr:
        print(item)
        np.savetxt(f, item, delimiter=',')
f.close()
