import sys, os

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

import lightgbm

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore

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

lgb = lightgbm.sklearn.LGBMClassifier(device='gpu')
met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
model = lgb.fit(data_list_dict["train_X"], data_list_dict["train_y"], eval_metric=met_grid, early_stopping_rounds=None,
                verbose=9)
from sklearn.metrics import roc_auc_score

print("model:", roc_auc_score(data_list_dict["test_y"], model.predict(data_list_dict["test_X"])))
print("lgb", roc_auc_score(data_list_dict["test_y"], lgb.predict(data_list_dict["test_X"])))

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
refit = "roc_auc"
lgb = lightgbm.sklearn.LGBMClassifier(device='gpu')
clf = RunAndScore(data_list_dict, lgb, parameters, met_grid, refit=refit, n_jobs=1)
writeRank2csv(met_grid, clf, cell_name, feature_name, method_name, dir_name)

print("clf.best_estimator_params:", clf.best_estimator_params_)
print("best params found in line [{1}] for metric [{0}] in rank file".format(refit, clf.best_estimator_params_idx_ + 2))

print(model.__dict__)
print(lgb.__dict__)
print(clf.get_best_estimator().__dict__)
print(model == lgb)
print(model == clf.get_best_estimator())
print(lgb == clf.get_best_estimator())
