import os
import sys
import time
import warnings
from itertools import product
from sklearnex import patch_sklearn

patch_sklearn()
start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from xgboost import XGBClassifier
from thundersvm import SVC
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from lightgbm.sklearn import LGBMClassifier

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since
from ML.EPIconst import EPIconst

estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier, "deepforest": CascadeForestClassifier, "lightgbm": LGBMClassifier}

names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[0]


def get_all_data(cell_name, all_feature_names):
    data_ensemble = {}
    for feature_name in all_feature_names:
        data_value = get_data_np_dict(cell_name, feature_name, EPIconst.MethodName.ensemble)
        data_ensemble.update({feature_name: data_value})
    return data_ensemble


def get_all_new_features(all_cell_names, all_feature_names, all_method_names):
    for item in product(all_cell_names, all_feature_names, all_method_names):
        print("ensemble_ex_item:", "_".join(item))
        model_params = getattr(EPIconst.Params, "_".join(item))
        cell_name = item[0]
        feature_name = item[1]
        method_name = item[2]
        estimator = estimators[method_name]()
        estimator.set_params(**model_params)
        print(estimator)
        print(estimator.get_params())

if __name__ == '__main__':
    # v = get_all_data(cell_name, EPIconst.FeatureName.all)
    # print(v)
    EPIconst.MethodName.all.remove("rf")
    get_all_new_features(EPIconst.CellName.all, EPIconst.FeatureName.all, EPIconst.MethodName.all)
