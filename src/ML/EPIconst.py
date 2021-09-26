class EPIconst:
    class FeatureName:
        pseknc = "pseknc"
        cksnap = "cksnap"
        dpcp = "dpcp"
        eiip = "eiip"
        kmer = "kmer"
        all = sorted([pseknc, cksnap, dpcp])

    class CellName:
        K562 = "K562"
        NHEK = "NHEK"
        IMR90 = "IMR90"
        HeLa_S3 = "HeLa-S3"
        HUVEC = "HUVEC"
        GM12878 = "GM12878"
        all = sorted([GM12878, HeLa_S3, HUVEC, IMR90, K562, NHEK])

    class MethodName:
        ensemble = "ensemble"
        xgboost = "xgboost"
        svm = "svm"
        deepforest = "deepforest"
        lightgbm = "lightgbm"
        rf = "rf"
        # all = sorted([xgboost, svm, deepforest, lightgbm, rf])
        all = sorted([lightgbm, rf])

    class BaseParams:
        xgboost = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                   'use_label_encoder': False, 'eval_metric': 'logloss', 'tree_method': 'gpu_hist'}
        svm = {"n_jobs": 1, "probability": True}
        lightgbm = {"n_jobs": 5, 'max_depth': -1, 'num_leaves': 31,
                    'min_child_samples': 20,
                    'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 0,
                    'reg_alpha': 0.0, 'reg_lambda': 0.0,
                    'min_split_gain': 0.0,
                    'objective': None,
                    'n_estimators': 100, 'learning_rate': 0.1,

                    'device': 'gpu', 'boosting_type': 'gbdt',
                    'class_weight': None, 'importance_type': 'split',
                    'min_child_weight': 0.001, 'random_state': None,
                    'subsample_for_bin': 200000, 'silent': True}
        rf = {"n_jobs": 5, 'n_estimators': 100, "max_depth": None, 'min_samples_split': 2, "min_samples_leaf": 1,
              'max_features': 'auto'}
        deepforest = {"n_jobs": 5, "use_predictor": False, "random_state": 1, "predictor": 'forest', "verbose": 0}

    class Params:
        GM12878_cksnap_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 250}
        GM12878_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 125, 'min_child_samples': 90,
                                   'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                   'reg_lambda': 1e-05, 'min_split_gain': 0.0, 'learning_rate': 0.1,
                                   'n_estimators': 250}
        GM12878_cksnap_svm = {}
        GM12878_cksnap_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 3, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 0,
                                  'learning_rate': 0.1}
        "----------------------------------------------"
        GM12878_dpcp_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        GM12878_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 331, 'max_bin': 135, 'min_child_samples': 190,
                                 'colsample_bytree': 0.7, 'subsample': 0.9, 'subsample_freq': 0, 'reg_alpha': 0.9,
                                 'reg_lambda': 0.001, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 250}
        GM12878_dpcp_svm = {}
        GM12878_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 3, 'reg_lambda': 3,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        GM12878_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 200}
        GM12878_pseknc_lightgbm = {'max_depth': 0, 'num_leaves': 291, 'max_bin': 145, 'min_child_samples': 150,
                                   'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 30, 'reg_alpha': 0.0,
                                   'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        GM12878_pseknc_svm = {}
        GM12878_pseknc_xgboost = {'n_estimators': 900, 'max_depth': 9, 'min_child_weight': 2, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                  'learning_rate': 0.07}
        "=============================================="
        HeLa_S3_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        HeLa_S3_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 341, 'max_bin': 105, 'min_child_samples': 80,
                                   'colsample_bytree': 0.9, 'subsample': 0.9, 'subsample_freq': 40, 'reg_alpha': 0.1,
                                   'reg_lambda': 0.1, 'min_split_gain': 0.4, 'learning_rate': 0.1, 'n_estimators': 150}
        HeLa_S3_cksnap_svm = {}
        HeLa_S3_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0,
                                  'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 3, 'reg_lambda': 0.5,
                                  'learning_rate': 0.1}
        "----------------------------------------------"
        HeLa_S3_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 400}
        HeLa_S3_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 221, 'max_bin': 155, 'min_child_samples': 180,
                                 'colsample_bytree': 0.7, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.2, 'learning_rate': 0.1, 'n_estimators': 200}
        HeLa_S3_dpcp_svm = {}
        HeLa_S3_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 3, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        HeLa_S3_pseknc_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        HeLa_S3_pseknc_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 5, 'min_child_samples': 110,
                                   'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                   'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        HeLa_S3_pseknc_svm = {}
        HeLa_S3_pseknc_xgboost = {'n_estimators': 850, 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                  'learning_rate': 0.1}
        "=============================================="
        HUVEC_cksnap_deepforest = {"max_layers": 10, "n_estimators": 10, "n_trees": 300}
        HUVEC_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 271, 'max_bin': 45, 'min_child_samples': 10,
                                 'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.5,
                                 'reg_lambda': 0.5, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 175}
        HUVEC_cksnap_svm = {}
        HUVEC_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 12, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.6, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        HUVEC_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 400}
        HUVEC_dpcp_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 245, 'min_child_samples': 30,
                               'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 50, 'reg_alpha': 0.5,
                               'reg_lambda': 0.3, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 200}
        HUVEC_dpcp_svm = {}
        HUVEC_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 3, 'reg_lambda': 3,
                              'learning_rate': 0.1}
        "----------------------------------------------"
        HUVEC_pseknc_deepforest = {'max_layers': 10, 'n_estimators': 2, 'n_trees': 250}
        HUVEC_pseknc_lightgbm = {'max_depth': 12, 'num_leaves': 261, 'max_bin': 235, 'min_child_samples': 110,
                                 'colsample_bytree': 0.9, 'subsample': 0.8, 'subsample_freq': 40, 'reg_alpha': 0.001,
                                 'reg_lambda': 0.001, 'min_split_gain': 1.0, 'learning_rate': 0.1, 'n_estimators': 225}
        HUVEC_pseknc_svm = {}
        HUVEC_pseknc_xgboost = {'n_estimators': 950, 'max_depth': 6, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "=============================================="
        IMR90_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 250}
        IMR90_cksnap_lightgbm = {'max_depth': 0, 'num_leaves': 271, 'max_bin': 95, 'min_child_samples': 60,
                                 'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.3, 'learning_rate': 0.1, 'n_estimators': 225}
        IMR90_cksnap_svm = {}
        IMR90_cksnap_xgboost = {'n_estimators': 900, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0.4,
                                'colsample_bytree': 0.6, 'subsample': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.1,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        IMR90_dpcp_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        IMR90_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 281, 'max_bin': 115, 'min_child_samples': 20,
                               'colsample_bytree': 0.7, 'subsample': 1.0, 'subsample_freq': 50, 'reg_alpha': 0.0,
                               'reg_lambda': 0.0, 'min_split_gain': 0.5, 'learning_rate': 0.1, 'n_estimators': 125}
        IMR90_dpcp_svm = {}
        IMR90_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 12, 'min_child_weight': 2, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.6, 'reg_alpha': 0.05, 'reg_lambda': 0.1,
                              'learning_rate': 0.1}
        "----------------------------------------------"
        IMR90_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 300}
        IMR90_pseknc_lightgbm = {'max_depth': -1, 'num_leaves': 321, 'max_bin': 45, 'min_child_samples': 30,
                                 'colsample_bytree': 0.9, 'subsample': 0.6, 'subsample_freq': 60, 'reg_alpha': 0.001,
                                 'reg_lambda': 0.0, 'min_split_gain': 0.9, 'learning_rate': 0.1, 'n_estimators': 100}
        IMR90_pseknc_svm = {}
        IMR90_pseknc_xgboost = {'n_estimators': 950, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "=============================================="
        K562_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 400}
        K562_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 311, 'max_bin': 225, 'min_child_samples': 60,
                                'colsample_bytree': 1.0, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.2, 'n_estimators': 250}
        K562_cksnap_svm = {}
        K562_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 6, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 2, 'reg_lambda': 0.05,
                               'learning_rate': 0.1}
        "----------------------------------------------"
        K562_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 300}
        K562_dpcp_lightgbm = {'colsample_bytree': 0.7, 'subsample': 0.7, 'subsample_freq': 80, 'reg_alpha': 1e-05,
                              'reg_lambda': 0.001, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 225}
        K562_dpcp_svm = {}
        K562_dpcp_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 4, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 1, 'reg_lambda': 0.05,
                             'learning_rate': 0.1}
        "----------------------------------------------"
        K562_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 250}
        K562_pseknc_lightgbm = {'max_depth': 0, 'num_leaves': 221, 'max_bin': 55, 'min_child_samples': 70,
                                'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 70, 'reg_alpha': 1e-05,
                                'reg_lambda': 1e-05, 'min_split_gain': 0.1, 'learning_rate': 0.1, 'n_estimators': 100}
        K562_pseknc_svm = {}
        K562_pseknc_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 5, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                               'learning_rate': 0.1}
        "=============================================="
        NHEK_cksnap_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 400}
        NHEK_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 291, 'max_bin': 205, 'min_child_samples': 90,
                                'colsample_bytree': 1.0, 'subsample': 0.9, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 75}
        NHEK_cksnap_svm = {}
        NHEK_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 2, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                               'learning_rate': 0.1}
        "----------------------------------------------"
        NHEK_dpcp_deepforest = {"max_layers": 10, "n_estimators": 8, "n_trees": 200}
        NHEK_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 301, 'max_bin': 145, 'min_child_samples': 70,
                              'colsample_bytree': 0.7, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.9,
                              'reg_lambda': 1.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 150}
        NHEK_dpcp_svm = {}
        NHEK_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 9, 'min_child_weight': 3, 'gamma': 0.5,
                             'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1,
                             'learning_rate': 0.1}
        "----------------------------------------------"
        NHEK_pseknc_deepforest = {"max_layers": 20, "n_estimators": 13, "n_trees": 400}
        NHEK_pseknc_lightgbm = {'max_depth': 12, 'num_leaves': 251, 'max_bin': 105, 'min_child_samples': 80,
                                'colsample_bytree': 0.7, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.7,
                                'reg_lambda': 0.7, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 125}
        NHEK_pseknc_svm = {}
        NHEK_pseknc_xgboost = {'n_estimators': 650, 'max_depth': 8, 'min_child_weight': 5, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.02,
                               'learning_rate': 0.1}


if __name__ == '__main__':
    print(getattr(EPIconst.Params, "NHEK_pseknc_deepforest"))
