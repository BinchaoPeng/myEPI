from ML.ml_def import get_scoring_result

y = [0, 1, 0, 0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 1, 1, 1]
y_prob = [0.7, 0.51, 0.4, 0.7, 0.3, 0.2, 0.71, 0.51, 0.31]
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']

for i in range(1, 10):
    print("####", i)
    get_scoring_result(sorted(met_grid), y, y_pred, y_prob)
