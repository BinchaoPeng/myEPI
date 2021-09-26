import os
import sys
import time
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sklearn.ensemble import RandomForestClassifier
from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[5]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'eiip', 'kmer', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[4]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm', 'rf']
method_name = method_names[4]
dir_name = "run_and_score"
ex_dir_name = '%s_%s_%s' % (feature_name, method_name, dir_name)
if not os.path.exists(r'../../ex/%s/' % ex_dir_name):
    os.mkdir(r'../../ex/%s/' % ex_dir_name)
    print("created ex folder!!!")
if not os.path.exists(r'../../ex/%s/rank' % ex_dir_name):
    os.mkdir(r'../../ex/%s/rank' % ex_dir_name)
    print("created rank folder!!!")


def rf_grid_greedy(cv_params, other_params, index):
    model = RandomForestClassifier(**other_params)
    print(model.get_params())
    refit = "roc_auc"
    met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
    clf = RunAndScore(data_list_dict, model, cv_params, met_grid, refit=refit, n_jobs=1, verbose=0)

    print("clf.best_estimator_params:", clf.best_estimator_params_)
    print("best params found in line [{1}] for metric [{0}] in rank file".format(refit,
                                                                                 clf.best_estimator_params_idx_ + 2))
    print("best params found in fit [{1}] for metric [{0}] in run_and_score file".format(refit,
                                                                                         clf.best_estimator_params_idx_ + 1))
    print("clf.best_scoring_result:", clf.best_scoring_result)

    writeRank2csv(met_grid, clf, cell_name, feature_name, method_name, dir_name, index)

    return clf.best_estimator_params_


best_params_result = {}
other_params = {'n_estimators': 100, "n_jobs": 5, "max_depth": None, 'min_samples_split': 2, "min_samples_leaf": 1,
                'max_features': 'auto'}

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

# 第一次：决策树的最佳数量也就是估计器的数目
print("第一次")
cv_params = {'n_estimators': list(range(10, 350, 10))}
# cv_params = {'n_estimators': [120]}
best_params = rf_grid_greedy(cv_params, other_params, '1')
other_params.update(best_params)
best_params_result.update(best_params)
# 第二次
print("第二次")
max_depth = [None]
max_depth.extend((list(range(1, 150))))
cv_params = {'max_depth': max_depth}
# cv_params = {'max_depth': [99]}
best_params = rf_grid_greedy(cv_params, other_params, '2')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第三次
print("第三次")
cv_params = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10], "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# cv_params = {'gamma': [0.1, 0.2, 0.3,]}
best_params = rf_grid_greedy(cv_params, other_params, '3')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第四次
print("第四次")
cv_params = {'max_features': ["auto", "sqrt", "log2", None]}
# cv_params = {'max_features': [0.6, 0.7, ]}
best_params = rf_grid_greedy(cv_params, other_params, '4')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

print("total time spending:", time_since(start_time))
print("best_params_result:", best_params_result)
"""
params
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 
'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 
'min_impurity_decrease': 0.0, 'min_impurity_split': None, 
'min_samples_leaf': 1, 'min_samples_split': 2, 
'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 
'n_jobs': 15, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

随机森林主要的参数有
n_estimators（子树的数量）、max_depth（树的最大生长深度）、min_samples_leaf（叶子的最小样本数量）、
min_samples_split(分支节点的最小样本数量）、max_features（最大选择特征数）


"""
