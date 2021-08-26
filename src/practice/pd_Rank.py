import pandas as pd

obj = pd.DataFrame({"mean": [7, -5, 7, 4, 2, 0, 4], "std": [1, 23, 456, 7, 7, 2, 5]})
print(obj)
# c = obj.rank(ascending=False, method="min")
# print(c.values.astype(int))

obj['ORD_RANK'] = obj.groupby(['mean', 'std']).ngroup(False) + 1
print(obj['ORD_RANK'])

import sklearn
print(sklearn.show_versions())