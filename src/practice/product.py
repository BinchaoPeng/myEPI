from itertools import product

k_l = range(1, 5)
for k in k_l:
    for i in product('ACGT', repeat=k):
        print(i)
