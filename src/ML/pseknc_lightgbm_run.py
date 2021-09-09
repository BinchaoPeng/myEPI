import sys, os
import copy

import lightgbm

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[1]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm']
method_name = method_names[3]
dir_name = "run_and_score"

"""
params
"""
parameters = [

    {

    },
]

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)
base_lgb = lightgbm.sklearn.LGBMClassifier(device='gpu')

lgb = copy.deepcopy(base_lgb)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
refit = "roc_auc"
clf = RunAndScore(data_list_dict, base_lgb, parameters, met_grid, refit=refit, n_jobs=1)
writeRank2csv(met_grid, clf, cell_name, feature_name, method_name, dir_name)

print("clf.best_estimator_params:", clf.best_estimator_params_)
print("best params found in fit [{1}] for metric [{0}] in rank file".format(refit, clf.best_estimator_params_idx_ + 1))
