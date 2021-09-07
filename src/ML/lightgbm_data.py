import lightgbm

from ML.ml_def import get_data_np_dict

import sys, os

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[1]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest']
method_name = method_names[2]
dir_name = "run_and_score"

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

lgb = lightgbm.sklearn.LGBMClassifier()
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
lgb.fit(data_list_dict["train_X"], data_list_dict["train_y"], eval_metric=met_grid, early_stopping_rounds=None,
        verbose=9)
y_pred = lgb.predict(data_list_dict["test_X"])
print(y_pred)
