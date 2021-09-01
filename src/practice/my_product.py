from itertools import product

candidate_params = ['a', 'b', 'c']
cv = [{'a': 1, 'b': 2}, 2, 3, 4, 5]
a = product(
    enumerate(candidate_params),
    enumerate(cv, 1))
for (cand_idx, parameters), (split_idx, test) in a:
    print(cand_idx, parameters, split_idx, test)

a = [{"1": 2}, {"2": 3}]
print(enumerate(a, 1))
for idx, item in enumerate(a, 1):
    print((idx, item))
