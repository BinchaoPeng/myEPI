import os
import sys
import time
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from xgboost import XGBClassifier
from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since

"""
cell and feature choose
"""
datasources = ['epivan', 'sept']
datasource = datasources[0]
names = ['pbc_IMR90', "GM12878", "HeLa-S3", "HMEC", "HUVEC", "IMR90", "K562", "NHEK", 'all', 'all-NHEK']
cell_name = names[6]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'eiip', 'kmer', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[4]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm']
method_name = method_names[1]
ensemble_steps = ["base", "meta"]
ensemble_step = ensemble_steps[0]
computers = ["2080ti", "3070", "3090"]
computer = computers[2]

ex_dir_name = '../../ex/%s/%s/%s_%s_%s' % (datasource, ensemble_step, feature_name, method_name, ensemble_step)
if not os.path.exists(ex_dir_name):
    os.mkdir(ex_dir_name)
    print(ex_dir_name, "created !!!")
if not os.path.exists(r'%s/rank' % ex_dir_name):
    os.mkdir(r'%s/rank' % ex_dir_name)
    print(ex_dir_name + "/rank", "created !!!")


def xgboost_grid_greedy(cv_params, other_params, index):
    model = XGBClassifier(**other_params)
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

    writeRank2csv(met_grid, clf, ex_dir_name, cell_name, computer, index=index)

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
best_params_result = {}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                'use_label_encoder': False, 'eval_metric': 'logloss', 'tree_method': 'gpu_hist'}

data_list_dict = get_data_np_dict(datasource, cell_name, feature_name, method_name)

# 第一次：决策树的最佳数量也就是估计器的数目
print("第一次")
cv_params = {'n_estimators': list(range(50, 1050, 50))}
# cv_params = {'n_estimators': list(range(50, 300, 50))}
best_params = xgboost_grid_greedy(cv_params, other_params, '1')
other_params.update(best_params)
best_params_result.update(best_params)

# 第二次
print("第二次")
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
# cv_params = {'max_depth': [3, 4, ], 'min_child_weight': [1, 2, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '2')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第三次
print("第三次")
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
# cv_params = {'gamma': [0.1, 0.2, 0.3,]}
best_params = xgboost_grid_greedy(cv_params, other_params, '3')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第四次
print("第四次")
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
# cv_params = {'subsample': [0.6, 0.7, ], 'colsample_bytree': [0.6, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '4')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第五次
print("第五次")
cv_params = {'reg_alpha': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3],
             'reg_lambda': [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3]}
# cv_params = {'reg_alpha': [0.05, ], 'reg_lambda': [0.05, 0.1, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '5')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

# 第六次
print("第六次")
cv_params = {'learning_rate': [0.001, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.75, 1.0]}
# cv_params = {'learning_rate': [0.01, 0.05, ]}
best_params = xgboost_grid_greedy(cv_params, other_params, '6')
other_params.update(best_params)
# print(other_params)
best_params_result.update(best_params)

print("total time spending:", time_since(start_time))
print("best_params_result:", best_params_result)
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
