from xgboost import XGBClassifier
from thundersvm import SVC
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
import lightgbm


module_pkgs = {"xgboost":XGBClassifier, "svm":SVC, "rf":RandomForestClassifier, "deepforest":CascadeForestClassifier, "lightgbm":lightgbm}

