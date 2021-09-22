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
feature_names = ['pseknc', 'cksnap', 'dpcp', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[2]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm']
method_name = method_names[1]
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


"""
params

alias:
{min_data_in_leaf}, default=20, type=int, alias=min_data_per_leaf , min_data, {min_child_samples}
一个叶子上数据的最小数量. 可以用来处理过拟合.

{bagging_fraction}, default=1.0, type=double, 0.0 &lt; bagging_fraction &lt; 1.0, alias=sub_row, {subsample}
类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
可以用来加速训练
可以用来处理过拟合
Note: 为了启用 bagging, bagging_freq 应该设置为非零值

{bagging_freq}, default=0, type=int, alias=subsample_freq
bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
Note: 为了启用 bagging, bagging_fraction 设置适当

{feature_fraction}, default=1.0, type=double, 0.0 &lt; feature_fraction &lt; 1.0, alias=sub_feature, {colsample_bytree}
如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征. 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征
可以用来加速训练
可以用来处理过拟合

"""
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                'use_label_encoder': False, 'eval_metric': 'logloss', 'tree_method': 'gpu_hist'}

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

# 第一次：决策树的最佳数量也就是估计器的数目
print("第一次")
cv_params = {'n_estimators': list(range(50, 1050, 50))}
# cv_params = {'n_estimators': list(range(50, 300, 50))}
best_params = rf_grid_greedy(cv_params, other_params, '1')
other_params.update(best_params)

# 第二次
print("第二次")
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
# cv_params = {'max_depth': [3, 4, ], 'min_child_weight': [1, 2, ]}
best_params = rf_grid_greedy(cv_params, other_params, '2')
other_params.update(best_params)
# print(other_params)

# 第三次
print("第三次")
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
# cv_params = {'gamma': [0.1, 0.2, 0.3,]}
best_params = rf_grid_greedy(cv_params, other_params, '3')
other_params.update(best_params)
# print(other_params)

# 第四次
print("第四次")
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
# cv_params = {'subsample': [0.6, 0.7, ], 'colsample_bytree': [0.6, ]}
best_params = rf_grid_greedy(cv_params, other_params, '4')
other_params.update(best_params)
# print(other_params)

# 第五次
print("第五次")
cv_params = {'reg_alpha': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3],
             'reg_lambda': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3]}
# cv_params = {'reg_alpha': [0.05, ], 'reg_lambda': [0.05, 0.1, ]}
best_params = rf_grid_greedy(cv_params, other_params, '5')
other_params.update(best_params)
# print(other_params)

# 第六次
print("第六次")
cv_params = {'learning_rate': [0.001, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.75, 1.0]}
# cv_params = {'learning_rate': [0.01, 0.05, ]}
best_params = rf_grid_greedy(cv_params, other_params, '6')
other_params.update(best_params)
# print(other_params)

print("total time spending:", time_since(start_time))

"""


"""
