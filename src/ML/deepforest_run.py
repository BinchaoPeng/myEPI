import sys, os
import warnings

warnings.filterwarnings("ignore")

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from deepforest import CascadeForestClassifier
from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since
import time

start_time = time.time()
"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[2]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[1]
method_names = ['svm', 'xgboost', 'deepforest']
method_name = method_names[2]
dir_name = "run_and_score"
ex_dir_name = '%s_%s_%s' % (feature_name, method_name, dir_name)
if not os.path.exists(r'../../ex/%s/' % ex_dir_name):
    os.mkdir(r'../../ex/%s/' % ex_dir_name)
    os.mkdir(r'../../ex/%s/rank' % ex_dir_name)
    print("created ex folder!!!")
"""
params
"""
parameters = [
    # {
    #     'n_estimators': [2, 5, 8, 10],
    #     'n_trees': [50, 100, 150, 200, 250, 300],
    #     'predictors': ['xgboost', 'lightgbm', 'forest'],
    #     'max_layers': [20, 50, 80, 120, 150],
    #     'use_predictor': [True]
    # },
    {
        'n_estimators': [2, 5, 8, 10, 13],
        'n_trees': [50, 100, 150, 200, 250, 300, 400],
        'max_layers': [20, 50, 80, 120, 150, 200],
    },
]
# parameters = [
#     {
#         'n_estimators': [2, 6],
#         # 'max_layers': [layer for layer in range(20, 40, 10)],
#         'predictors': ['xgboost', 'lightgbm', 'forest'],
#         'use_predictor': [True]
#     },
# ]

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)
deep_forest = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest', verbose=0)

# import xgboost as xgb
# import lightgbm as lgb
# # set estimator
# n_estimators = 4  # the number of base estimators per cascade layer
# estimators = [xgb.XGBClassifier(tree_method="gpu_hist", random_state=1) for i in range(n_estimators)]
# deep_forest.set_estimator(estimators)
#
# # set predictor
# predictor = xgb.XGBClassifier(tree_method="gpu_hist", eval_metric="error", random_state=1)
# deep_forest.set_predictor(predictor)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
refit = "roc_auc"
clf = RunAndScore(data_list_dict, deep_forest, parameters, met_grid, refit=refit, n_jobs=3)
writeRank2csv(met_grid, clf, cell_name, feature_name, method_name, dir_name)

print("clf.best_estimator_params:", clf.best_estimator_params_)
print("best params found in line [{1}] for metric [{0}] in rank file".format(refit, clf.best_estimator_params_idx_ + 2))
print("total time spending:", time_since(start_time))
"""
param doc
{'n_bins': 255, 'bin_subsample': 200000, 'bin_type': 'percentile', 'max_layers': 20, 
'criterion': 'gini', 'n_estimators': 2, 'n_trees': 100, 'max_depth': None, 
'min_samples_leaf': 1, 'predictor_kwargs': {}, 'backend': 'custom', 'n_tolerant_rounds': 2, 
'delta': 1e-05, 'partial_mode': False, 'n_jobs': 5, 'random_state': 1, 'verbose': 0, 'n_layers_': 0, 
'is_fitted_': False, 'layers_': {}, 'binners_': {}, 
'buffer_': <deepforest._io.Buffer object at 0x7f84cb1adc50>, 'use_predictor': False, 
'predictor': 'forest', 'labels_are_encoded': False, 'type_of_target_': None, 'label_encoder_': None}

一、更好的准确性
决定是否添加预测器的一个有用规则是将深度森林的性能与从训练数据生成的独立预测器的性能进行比较。
如果预测器始终优于深度森林，那么通过添加预测器，深度森林的性能有望得到改善。
在这种情况下，从深森林产生的增强特征也有助于训练预测器。
1. 增加模型复杂性
n_estimators：指定每个级联层中的估计器数量。
n_trees：指定每个估计器中的树数。
max_layers：指定最大级联层数。
使用上述较大的参数值，深度森林的性能可能会提高复杂数据集的性能，这些数据集需要更大的模型才能表现良好。

2. 添加预测器
use_predictor：决定是否使用连接到深森林的预测器。
predictor: 指定预测器的类型，应为"forest", "xgboost", "lightgbm" 之一


二、更快的速度
由于深度森林根据训练数据的验证性能自动确定模型复杂度，因此将参数设置为较小的值可能会导致具有更多级联层的深度森林模型。
1. 并行化
强烈建议使用并行化，因为深森林自然适合它。
n_jobs：指定使用的工人数量。将其值设置为大于 1 的整数将启用并行化。将其值设置为-1意味着使用所有处理器。

2. 更少的分裂
n_bins：指定特征离散 bin 的数量。较小的值意味着将考虑较少的分裂截止值，应为 [2, 255] 范围内的整数。
bin_type：指定分箱类型。将其值设置为"interval"可以在特征值累积的密集间隔上考虑较少的分割截止点。

3. 降低模型复杂度
将以下参数设置为较小的值会降低深度森林的模型复杂度，并且可能会导致更快的训练和评估速度。
max_depth: 指定树的最大深度。None表示没有约束。
min_samples_leaf：指定叶节点所需的最小样本数。最小值为1。
n_estimators：指定每个级联层中的估计器数量。
n_trees：指定每个估计器中的树数。
n_tolerant_rounds: 指定处理早停时的容忍轮数。最小值为1。

三、降低内存使用率
1. 部分模式
partial_mode：决定是否在部分模式下训练和评估模型。如果设置为True，模型将主动将拟合的估计量转储到本地缓冲区中。
因此，深度森林的内存使用不再随着拟合级联层数的增加而线性增加。

此外，降低模型复杂度也会降低内存使用量。

"""
