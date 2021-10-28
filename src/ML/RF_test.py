import os
import sys
import time
import warnings

from ML.EPIconst import EPIconst

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sklearn.ensemble import RandomForestClassifier
from ML.ml_def import get_data_np_dict, time_since, get_scoring_result

cell_name = EPIconst.CellName.GM12878
feature_names = ["psednc_II_lam3_w1", "psednc_II_lam4_w1", "psednc_II_lam5_w1", "psednc_II_lam6_w1",
                 "psetnc_II_lam3_w1", "psetnc_II_lam4_w1", "psetnc_II_lam5_w1", "psetnc_II_lam20_w1",
                 "psetnc_II_lam40_w1"]
method_name = EPIconst.MethodName.rf

for feature_name in feature_names:
    data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

    met_grid = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])

    clf = RandomForestClassifier(n_jobs=os.cpu_count() - 2)
    clf.fit(data_list_dict['train_X'], data_list_dict['train_y'])
    y_pred = clf.predict(data_list_dict['test_X'])
    y_pred_prob_temp = clf.predict_proba(data_list_dict['test_X'])
    if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
        y_proba = y_pred_prob_temp[:, 0]
    else:
        y_proba = y_pred_prob_temp[:, 1]

    process_msg, score_result_dict = get_scoring_result(met_grid, data_list_dict["test_y"], y_pred, y_proba)
    print("{0}_{1}: {2} End time={3}\n".format(cell_name, feature_name, process_msg, time_since(start_time)))
