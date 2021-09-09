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


def lgb_grid_greedy(cv_params, other_params, index):
    base_lgb = lightgbm.sklearn.LGBMClassifier(device='gpu')
    base_lgb.set_params(other_params)

    met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
    clf = RunAndScore(data_list_dict, base_lgb, cv_params, met_grid, refit=refit, n_jobs=1)

    print("clf.best_estimator_params:", clf.best_estimator_params_)
    print("best params found in fit [{1}] for metric [{0}] in rank file".format(refit,
                                                                                clf.best_estimator_params_idx_ + 1))
    writeRank2csv(met_grid, clf, index)

    return clf.best_estimator_params_


"""
params
"""
other_params = [
    {'num_leaves': 31, 'objective': None, 'learning_rate': 0.1, 'max_depth': -1, 'reg_alpha': 0.0, 'reg_lambda': 0.0,
     'n_estimators': 100, 'boosting_type': 'gbdt',
     'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split',

     'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_jobs': -1, 'random_state': None,

     'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'device': 'gpu', 'silent': True}
]
parameters = [

    {

    },
]

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)
base_lgb = lightgbm.sklearn.LGBMClassifier(device='gpu')
"""
{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 
'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 31, 
'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True, 
'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'device': 'gpu'}
"""
print("p:", base_lgb.get_params())
lgb = copy.deepcopy(base_lgb)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
refit = "roc_auc"
clf = RunAndScore(data_list_dict, base_lgb, parameters, met_grid, refit=refit, n_jobs=1)
writeRank2csv(met_grid, clf, cell_name, feature_name, method_name, dir_name)

print("clf.best_estimator_params:", clf.best_estimator_params_)
print("best params found in fit [{1}] for metric [{0}] in rank file".format(refit, clf.best_estimator_params_idx_ + 1))

"""
### 针对 Leaf-wise (最佳优先) 树的参数优化

1. `num_leaves`

   控制树模型复杂度的主要参数。应让其小于`2^(max_depth)`，因为`depth` 的概念在 leaf-wise 树中并没有多大作用，并不存在从`leaves`到`depth`的映射

2. `min_data_in_leaf`

   用于处理过拟合，该值取决于训练样本数和`num_leaves`，几百或几千即可。设置较大避免生成一个过深的树，可能导致欠拟合。

3. `max_depth`

   显示限制树的深度

### 针对更快的训练速度

- 通过设置 `bagging_fraction` 和 `bagging_freq` 参数来使用 bagging 方法
- 通过设置 `feature_fraction` 参数来使用特征的子抽样
- 使用较小的 `max_bin`
- 使用 `save_binary` 在未来的学习过程对数据加载进行加速
- 使用并行学习, 可参考 [并行学习指南](https://www.kancloud.cn/apachecn/lightgbm-doc-zh/Parallel-Learning-Guide.rst)

### 针对更好的准确率

- 使用较大的 `max_bin` （学习速度可能变慢）
- 使用较小的 `learning_rate` 和较大的 `num_iterations`
- 使用较大的 `num_leaves` （可能导致过拟合）
- 使用更大的训练数据
- 尝试 `dart`

### 处理过拟合

- 使用较小的 `max_bin`
- 使用较小的 `num_leaves`
- 使用 `min_data_in_leaf` 和 `min_sum_hessian_in_leaf`
- 通过设置 `bagging_fraction` 和 `bagging_freq` 来使用 bagging
- 通过设置 `feature_fraction` 来使用特征子抽样
- 使用更大的训练数据
- 使用 `lambda_l1`, `lambda_l2` 和 `min_gain_to_split` 来使用正则
- 尝试 `max_depth` 来避免生成过深的树


"""
