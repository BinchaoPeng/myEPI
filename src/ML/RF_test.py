import os
import sys
import time
import warnings
# from sklearnex import patch_sklearn
#
# patch_sklearn()
from ML.EPIconst import EPIconst

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sklearn.ensemble import RandomForestClassifier
from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since, get_scoring_result

cell_name = EPIconst.CellName.HeLa_S3
feature_name = EPIconst.FeatureName.kmer
method_name = EPIconst.MethodName.rf
data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

met_grid = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])
cell_name_temp = cell_name + "_" + feature_name + "_" + method_name
if cell_name.__contains__("HeLa-S3"):
    cell_name_temp = "HeLa_S3" + "_" + feature_name + "_" + method_name
test_params = getattr(EPIconst.ModelParams, cell_name_temp)
# print(test_params)

clf = RandomForestClassifier(**EPIconst.ModelBaseParams.rf)
# print(clf.get_params())
clf.set_params(**EPIconst.ModelBaseParams.rf)
clf.set_params(**test_params)
print("clf:", clf)
clf.fit(data_list_dict['train_X'], data_list_dict['train_y'])
y_pred = clf.predict(data_list_dict['test_X'])
y_pred_prob_temp = clf.predict_proba(data_list_dict['test_X'])
if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
    y_proba = y_pred_prob_temp[:, 0]
else:
    y_proba = y_pred_prob_temp[:, 1]

print("y_pred:", y_pred[0:20])
print("y_proba_temp:", y_pred_prob_temp[0:20, :])
print("y_proba:", y_proba[0:20])
print("len y_pred:", len(y_pred))
print("len y_proba_temp:", len(y_pred_prob_temp))
print("len y_proba:", len(y_proba))
process_msg, score_result_dict = get_scoring_result(met_grid, data_list_dict["test_y"], y_pred, y_proba)
print("{0}_{1}: {2} End time={3}\n".format(cell_name, feature_name, process_msg, time_since(start_time)))
