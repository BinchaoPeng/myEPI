# In[ ]:

import numpy as np
import sys

sys.path.extend(['/home/pbc/Documents/PycharmProjects/myEPI/src/practice'])
"""
write 3-dim np_arr to csv
"""
l = [[[1, 1, 1],
      [1, 1, 1]],
     [[2, 2, 2],
      [2, 2, 2]]]

arr = np.array(l)
with open(r"../dataUtils/test1.csv", 'ab') as f:

    for item in arr:
        print(item)
        np.savetxt(f, item, delimiter=',')
f.close()