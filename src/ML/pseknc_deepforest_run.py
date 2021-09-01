from deepforest import CascadeForestClassifier
from ml_def import get_data_np_dict, get_score_dict, get_scoring_result, writeRank2csv, RunAndScore

"""
cell and feature choose
"""
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[1]
feature_names = ['pseknc', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[0]
method_names = ['svm', 'xgboost', 'deepforest']
method_name = method_names[2]

"""
params
"""
parameters = [
    {
        'n_estimators': [2, 5, 8],
        'n_trees': [50, 100, 150, 200],
        'predictors': ['xgboost', 'lightgbm', 'forest'],
        'max_layers': [20, 50, 80],
        'use_predictor': [True]
    },
    {
        'n_estimators': [2, 5, 8],
        'n_trees': [50, 100, 150, 200],
        'max_layers': [20, 50, 80]
    },
]
parameters = [

    {
        'n_estimators': [2, 4],
        # 'max_layers': [layer for layer in range(20, 40, 10)],
        'predictors': ['xgboost', 'lightgbm', 'forest'],
        'use_predictor': [True]
    },

]

data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)

deep_forest = CascadeForestClassifier(use_predictor=False, random_state=1, n_jobs=5, predictor='forest', verbose=0)

met_grid = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']

clf = RunAndScore(data_list_dict, deep_forest, parameters, met_grid, refit="roc_auc", n_jobs=2)
writeRank2csv(met_grid, clf, "fit_and_score")
