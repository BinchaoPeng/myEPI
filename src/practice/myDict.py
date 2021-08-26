score_dict = {'a': 2, 'd': 1, 'b': 2}
score_dict = sorted(score_dict.items(), key=lambda x: x[0], reverse=False)
score_dict = dict(score_dict)
print(score_dict)