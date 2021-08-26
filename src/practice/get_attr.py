from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from inspect import signature

train_y = [1, 1, 0, 0, 1]
test_y = [1, 1, 1, 0, 1]

# met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy']
# # accuracy_score = getattr(metrics, 'accuracy_score')
# module_name = __import__("sklearn.metrics", fromlist='*')
# print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
# accuracy_score = getattr(module_name, 'accuracy_score')
# score = accuracy_score(train_y, test_y)
# print(score)
#
# print('accuracy' in met_grid)
megrid = ['f1', 'roc_auc', 'average_precision', 'accuracy']


def get_scoring_result(scoring, y, y_pred, y_prob, y_score=None):
    if y_score is None:
        y_score = y_prob
    module_name = __import__("sklearn.metrics", fromlist='*')
    # print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
    score_dict = {}
    if isinstance(scoring, str):
        score_dict.update({scoring + '_score': scoring})
    else:
        for item in scoring:
            score_dict.update({item + '_score': item})
    # print(score_dict)
    # start get_scoring_result
    if 'accuracy' in score_dict.values():
        score_dict.pop('accuracy_score')
        score = getattr(module_name, 'accuracy_score')
        score_result = score(y, y_pred)
        # accuracy: (test=0.926)
        print("'accuracy': (test=%s) " % (score_result))

    for k, v in score_dict.items():
        score = getattr(module_name, k)
        sig = signature(score)
        y_flag = str(list(sig.parameters.keys())[1])
        if y_flag == 'y_pred':
            y_flag = y_pred

        elif y_flag == 'y_prob':
            y_flag = y_prob

        elif y_flag == 'y_score':
            y_flag = y_score
        else:
            raise ValueError("")
        if y_flag is None:
            raise ValueError(k, "%s is None" % (y_flag))
        score_result = score(y, y_flag)
        # accuracy: (test=0.926)
        print("%s: (test=%s) " % (v, score_result))


y = [1, 1, 0, 0, 1]
y_pred = [1, 1, 1, 0, 1]
y_pred_prob = [0.8, 0.51, 0.61, 0.4, 0.91]
get_scoring_result(megrid, y, y_pred, y_pred_prob, y_score=None)
