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
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # , StackingClassifier
from deepforest import CascadeForestClassifier
from lightgbm.sklearn import LGBMClassifier
from mlxtend.classifier import StackingClassifier
from ML.ml_def import get_data_np_dict, writeRank2csv, RunAndScore, time_since, get_scoring_result
from ML.EPIconst import EPIconst

estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier, "deepforest": CascadeForestClassifier,
              "lightgbm": LGBMClassifier}

cell_names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = cell_names[1]
feature_names = ['pseknc', 'cksnap', 'dpcp', 'eiip', 'kmer', 'dnabert_6mer', 'longformer-hug', 'elmo']
feature_name = feature_names[4]
EPIconst.MethodName.all.remove("deepforest")
for feature_item in EPIconst.FeatureName.all:
    start_time = time.time()
    """
    init base estimators
    """
    feature_names = [feature_item]

    all_method_names = EPIconst.MethodName.all
    base_models = []
    for item in product(feature_names, all_method_names):
        ex_item = cell_name + "_" + "_".join(item)
        feature = item[0]
        method_name = item[1]
        if ex_item.__contains__("HeLa-S3"):
            ex_item = "HeLa_S3" + "_" + feature + "_" + method_name
        model_params = getattr(EPIconst.ModelParams_epivan, ex_item)
        base_params = getattr(EPIconst.ModelBaseParams, method_name)
        estimator = estimators[method_name]()
        estimator.set_params(**base_params)
        estimator.set_params(**model_params)
        # base_models.append((ex_item, estimator))
        base_models.append(estimator)
        print(ex_item, ":", estimator)

    print(len(base_models))

    """
    set stacking
    """
    dataList = get_data_np_dict(cell_name, feature_name, EPIconst.MethodName.ensemble)
    # clf = StackingClassifier(estimators=base_models, final_estimator=None, cv=None, stack_method='auto', n_jobs=2,
    #                          passthrough=False)
    clf = StackingClassifier(classifiers=base_models, meta_classifier=LogisticRegression(), use_probas=False)
    clf.fit(dataList['train_X'], dataList['train_y'])
    y_pred = clf.predict(dataList['test_X'])
    y_pred_prob_temp = clf.predict_proba(dataList['test_X'])
    if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
        y_proba = y_pred_prob_temp[:, 0]
    else:
        y_proba = y_pred_prob_temp[:, 1]
    scoring = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])
    process_msg, score_result_dict = get_scoring_result(scoring, dataList["test_y"], y_pred, y_proba)
    print("{0}_{1}: {2} End time={3}\n".format(cell_name, feature_name, process_msg, time_since(start_time)))
