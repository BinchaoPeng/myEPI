import itertools
import math
import numpy as np

# itertools
a = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
out = list(itertools.chain.from_iterable(a))
# print(out)

# generator
c = [math.pow(2, i) for i in range(-12, 12)]
# print(c)

# var to list
v = 'ddd'
print(type(v),
      isinstance(v, str))
a = ["a", "b"]
print(type(a), isinstance(a, list))

# 2-dim

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[:, 1])
