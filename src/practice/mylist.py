import itertools

a = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
out = list(itertools.chain.from_iterable(a))
print(out)
