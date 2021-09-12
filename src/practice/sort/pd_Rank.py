import pandas as pd

obj = pd.DataFrame({"mean": [7, -5, 7, 4, 2, 0, 4], "std": [1, 23, 456, 7, 7, 2, 5]})
# print(obj)

obj['ORD_RANK'] = obj.groupby(['mean', 'std']).ngroup(False) + 1
# print(obj['ORD_RANK'])


cv_results = {"a":[97.123, 94.235, 78.456, 97.123, 92.327],
             "b":[97.123, 94.235, 78.456, 97.123, 92.327],
             "c":[97.123, 94.235, 78.456, 97.123, 92.327]}
scoring = ["a","b","c"]

for item in scoring:
    # sorted by mean
    obj = pd.Series(cv_results[item])
    c = obj.rank(ascending=False, method="min")
    # print(c.values.astype(int))
    cv_results.update({"rank_test_%s" % item: c.values.astype(int)})
    print(cv_results)



