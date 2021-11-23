import os
import sys
import time
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sklearn.neural_network import MLPClassifier

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since

"""
cell and feature choose
"""
datasources = ['epivan', 'sept']
datasource = datasources[0]
names = ['pbc_IMR90', "GM12878", "HeLa-S3", "HMEC", "HUVEC", "IMR90", "K562", "NHEK", 'all', 'all-NHEK']
cell_name = names[1]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[1]
method_names = ['svm', 'xgboost', 'deepforest', 'lightgbm', 'MLP']
method_name = method_names[4]
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

"""
params

{'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'batch_size': 'auto', 'learning_rate': 'constant', 
'learning_rate_init': 0.001, 'power_t': 0.5, 'max_iter': 200, 'loss': 'log_loss', 'hidden_layer_sizes': (100,), 
'shuffle': True, 'random_state': None, 'tol': 0.0001, 'verbose': False, 'warm_start': False, 'momentum': 0.9,
 'nesterovs_momentum': True, 'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 
 'epsilon': 1e-08, 'n_iter_no_change': 10, 'max_fun': 15000}

"""
parameters = [
    # {'activation': ['relu', 'identity', 'logistic', 'tanh'], 'solver': ["lbfgs", ],
    #  'alpha': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    #  'max_iter': [200, 300], 'loss': ['log_loss'],
    #  'hidden_layer_sizes': (100,),
    #  'random_state': [None], 'tol': [0.0001],
    #  'n_iter_no_change': [10], 'max_fun': [15000]},
    {'activation': ['relu', 'identity', 'logistic', 'tanh'], 'solver': ['adam', ],
     'alpha': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'batch_size': [64, 128, 200, 256, 512, 1024],
     'learning_rate_init': [0.0001, 0.001],
     'max_iter': [200], 'loss': ['log_loss'],
     'hidden_layer_sizes': (100,),
     'shuffle': [False], 'random_state': [None], 'tol': [0.0001],
     'early_stopping': [False], 'validation_fraction': [0.1], 'beta_1': [0.9],
     'beta_2': [0.999],
     'epsilon': [1e-08], 'n_iter_no_change': [10], },
    {'activation': ['relu', 'identity', 'logistic', 'tanh'], 'solver': ['"sgd"', ],
     'alpha': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'batch_size': [64, 128, 200, 256, 512, 1024],
     'learning_rate': ['adaptive'], 'learning_rate_init': [0.0001, 0.001],
     'power_t': [0.1, 0.3, 0.5, 0.7, 0.9], 'max_iter': [200, 300], 'loss': ['log_loss'],
     'hidden_layer_sizes': (100,),
     'shuffle': [True], 'random_state': [None], 'tol': [0.0001],
     'momentum': [0.9],
     'nesterovs_momentum': [True], 'early_stopping': [True], 'validation_fraction': [0.1],
     'n_iter_no_change': [10], }
]

data_list_dict = get_data_np_dict(datasource, cell_name, feature_name, method_name)
mlp = MLPClassifier()  # 调参
print(mlp.__dict__)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
refit = "roc_auc"
clf = RunAndScore(data_list_dict, mlp, parameters, met_grid, refit=refit, n_jobs=1)
writeRank2csv(met_grid, clf, ex_dir_name, cell_name, computer)

print("clf.best_estimator_params:", clf.best_estimator_params_)
print("best params found in line [{1}] for metric [{0}] in rank file".format(refit, clf.best_estimator_params_idx_ + 2))
print("best params found in fit [{1}] for metric [{0}] in run_and_score file".format(refit,
                                                                                     clf.best_estimator_params_idx_ + 1))
print("clf.best_scoring_result:", clf.best_scoring_result)
print("total time spending:", time_since(start_time))
