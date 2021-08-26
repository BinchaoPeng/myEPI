import pandas as pd
import numpy as np
import copy

all_out = [[{'max_layers': 20, 'n_estimators': 2},
            {'accuracy': 0.7269928966061563, 'average_precision': 0.986664818258594, 'f1': 0.6253249566724437,
             'roc_auc': 0.9875328881795665}],
           [{'max_layers': 20, 'n_estimators': 2},
            {'accuracy': 0.9814127861089187, 'average_precision': 0.9998716142341333,
             'f1': 0.9810820580792866, 'roc_auc': 0.9998755643847085}],
           [{'max_layers': 20, 'n_estimators': 2},
            {'accuracy': 0.781136543014996, 'average_precision': 0.9922351758132908, 'f1': 0.7202945329836595,
             'roc_auc': 0.9919860112328754}],
           [{'max_layers': 30, 'n_estimators': 2},
            {'accuracy': 0.7269928966061563, 'average_precision': 0.986664818258594,
             'f1': 0.6253249566724437, 'roc_auc': 0.9875328881795665}],
           [{'max_layers': 30, 'n_estimators': 2},
            {'accuracy': 0.9814127861089187, 'average_precision': 0.9998716142341333, 'f1': 0.9810820580792866,
             'roc_auc': 0.9998755643847085}],
           [{'max_layers': 30, 'n_estimators': 2},
            {'accuracy': 0.781136543014996, 'average_precision': 0.9922351758132908,
             'f1': 0.7202945329836595, 'roc_auc': 0.9919860112328754}]]

cv = 3
scoring = ['f1', 'roc_auc', 'average_precision', 'accuracy']


def _get_score_dict():
    score_dict = {}
    if isinstance(scoring, str):
        score_dict.update({scoring + '_score': scoring})
    else:
        for item in scoring:
            score_dict.update({item + '_score': item})
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
    print(score_dict)
    return score_dict


def get_cv_results():
    cv_results = {}
    cv_score_temp = {}
    # define attr of cv_results and cv_score_temp
    score_dict = _get_score_dict()
    for k, v in score_dict.items():
        cv_results.update({"rank_test_%s" % v: [], "mean_test_%s" % v: [], "std_test_%s" % v: [], })
        cv_score_temp.update({v: np.array([])})
    cv_results.update({"params": []})

    # set data into cv_results
    n = cv
    for i in range(0, len(all_out), n):
        cv_out = all_out[i:i + n]
        cv_results["params"].append(cv_out[0][0])
        cv_out_score_temp = copy.deepcopy(cv_score_temp)
        print("temp:", cv_out_score_temp)
        for idx in range(0, cv):
            # set mean_test and std_test for per cv
            cv_out_score = cv_out[idx][1]
            for k, v in score_dict.items():
                cv_out_score_temp[v] = np.append(cv_out_score_temp[v], cv_out_score[v])
        print(cv_out_score_temp)
        for k, v in cv_out_score_temp.items():
            cv_results["mean_test_%s" % k].append(v.mean())
            cv_results["std_test_%s" % k].append(v.std())
    print(cv_results)

    obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
    c = obj.rank(ascending=False, method="min")
    print(c.values.astype(int))

    return cv_results


get_cv_results()
