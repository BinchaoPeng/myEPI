import numpy as np

score_dict = {'a': 2, 'd': 1, 'b': 2}
score_dict = sorted(score_dict.items(), key=lambda x: x[0], reverse=False)
score_dict = dict(score_dict)
print(score_dict)

ATGC_base = {"AA": 0, "AC": 0, "AG": 0, "AT": 0,
             "CA": 0, "CC": 0, "CG": 0, "CT": 0,
             "GA": 0, "GC": 0, "GG": 0, "GT": 0,
             "TA": 0, "TC": 0, "TG": 0, "TT": 0}

b = ATGC_base.copy()

b.clear()

print(b)
print(ATGC_base)

score_dict = {"A": {'a': 2, 'd': 1, 'b': 2}}
print(score_dict["A"]["d"])

li = ['a', 'b']
l = np.zeros(len(li))
print(l)
print(dict(zip(li, l)))
