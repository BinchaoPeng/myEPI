import itertools
import math
# itertools
a = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
out = list(itertools.chain.from_iterable(a))
print(out)

# generator
c = [math.pow(2, i) for i in range(-12, 12)]
print(c)