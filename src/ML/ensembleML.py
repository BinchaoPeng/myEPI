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
from ML.EPIconst import EPIconst

names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
cell_name = names[6]


def get_all_data(cell_name, all_feature):
    data_ensemble = {}
    for feature_name in all_feature:
        data_value = get_data_np_dict(cell_name, feature_name, EPIconst.MethodName.ensemble)
        data_ensemble.update({feature_name: data_value})
    return data_ensemble


if __name__ == '__main__':
    v = get_all_data(cell_name, EPIconst.FeatureName.all)
    print(v)
