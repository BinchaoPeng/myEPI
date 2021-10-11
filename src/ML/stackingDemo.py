import os
import sys
import time
import warnings
from itertools import product
import numpy as np
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

from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since, get_scoring_result
from ML.EPIconst import EPIconst

estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier, "deepforest": CascadeForestClassifier,
              "lightgbm": LGBMClassifier}
method_name = "svm"
meta_estimator = estimators[method_name]()
base_params = getattr(EPIconst.ModelBaseParams, method_name)
meta_estimator.set_params(**base_params)
svc = SVC()
svc.set_params(**base_params)
