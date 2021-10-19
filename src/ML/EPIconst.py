class EPIconst:
    class FeatureName:
        pseknc = "pseknc"
        cksnap = "cksnap"
        dpcp = "dpcp"
        eiip = "eiip"
        kmer = "kmer"
        all = sorted([pseknc, cksnap, dpcp, eiip, kmer])

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
        all = sorted([lightgbm, rf, xgboost, svm, deepforest, ])

    class ModelBaseParams:
        deepforest = {"n_jobs": 16, "use_predictor": False, "random_state": 1, "predictor": 'forest', "verbose": 0}
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
        svm = {"n_jobs": 5, "probability": True}
        # svm = {"probability": True}
        xgboost = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                   'use_label_encoder': False, 'eval_metric': 'logloss', 'tree_method': 'gpu_hist'}

    class ModelParams:
        GM12878_cksnap_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 250}
        GM12878_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 125, 'min_child_samples': 90,
                                   'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                   'reg_lambda': 1e-05, 'min_split_gain': 0.0, 'learning_rate': 0.1,
                                   'n_estimators': 250}
        GM12878_cksnap_svm = {'C': 4.0, 'gamma': 64.0, 'kernel': 'rbf'}
        GM12878_cksnap_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 3, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 0,
                                  'learning_rate': 0.1}
        GM12878_cksnap_rf = {'n_estimators': 340, 'max_depth': 114, 'min_samples_leaf': 3, 'min_samples_split': 2,
                             'max_features': 'sqrt'}
        "----------------------------------------------"
        GM12878_dpcp_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        GM12878_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 331, 'max_bin': 135, 'min_child_samples': 190,
                                 'colsample_bytree': 0.7, 'subsample': 0.9, 'subsample_freq': 0, 'reg_alpha': 0.9,
                                 'reg_lambda': 0.001, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 250}
        GM12878_dpcp_svm = {'C': 1.0, 'gamma': 64.0, 'kernel': 'rbf'}
        GM12878_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 3, 'reg_lambda': 3,
                                'learning_rate': 0.1}
        GM12878_dpcp_rf = {'n_estimators': 150, 'max_depth': 88, 'min_samples_leaf': 1, 'min_samples_split': 3,
                           'max_features': None}
        "----------------------------------------------"
        GM12878_eiip_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        GM12878_eiip_lightgbm = {'max_depth': 12, 'num_leaves': 291, 'max_bin': 115, 'min_child_samples': 40,
                                 'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 50, 'reg_alpha': 1e-05,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        GM12878_eiip_rf = {'n_estimators': 280, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 7,
                           'max_features': None}
        GM12878_eiip_svm = {'C': 1.0, 'gamma': 2048.0, 'kernel': 'rbf'}
        GM12878_eiip_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 6, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        GM12878_kmer_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        GM12878_kmer_lightgbm = {'max_depth': 12, 'num_leaves': 291, 'max_bin': 115, 'min_child_samples': 40,
                                 'colsample_bytree': 1.0, 'subsample': 0.8, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        GM12878_kmer_rf = {'n_estimators': 170, 'max_depth': 41, 'min_samples_leaf': 3, 'min_samples_split': 2,
                           'max_features': None}
        GM12878_kmer_svm = {'C': 64.0, 'gamma': 16.0, 'kernel': 'rbf'}
        GM12878_kmer_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 6, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}
        "----------------------------------------------"
        GM12878_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 200}
        GM12878_pseknc_lightgbm = {'max_depth': 0, 'num_leaves': 291, 'max_bin': 145, 'min_child_samples': 150,
                                   'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 30, 'reg_alpha': 0.0,
                                   'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        GM12878_pseknc_rf = {'n_estimators': 130, 'max_depth': 54, 'min_samples_leaf': 1, 'min_samples_split': 6,
                             'max_features': 'sqrt'}
        GM12878_pseknc_svm = {'C': 0.5, 'gamma': 1024.0, 'kernel': 'rbf'}
        GM12878_pseknc_xgboost = {'n_estimators': 900, 'max_depth': 9, 'min_child_weight': 2, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                  'learning_rate': 0.07}

        "=============================================="
        HeLa_S3_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        HeLa_S3_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 341, 'max_bin': 105, 'min_child_samples': 80,
                                   'colsample_bytree': 0.9, 'subsample': 0.9, 'subsample_freq': 40, 'reg_alpha': 0.1,
                                   'reg_lambda': 0.1, 'min_split_gain': 0.4, 'learning_rate': 0.1, 'n_estimators': 150}
        HeLa_S3_cksnap_svm = {'C': 512.0, 'gamma': 16.0, 'kernel': 'rbf'}
        HeLa_S3_cksnap_rf = {'n_estimators': 340, 'max_depth': 44, 'min_samples_leaf': 1, 'min_samples_split': 5,
                             'max_features': 'sqrt'}
        HeLa_S3_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 4, 'gamma': 0,
                                  'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 3, 'reg_lambda': 0.5,
                                  'learning_rate': 0.1}

        "----------------------------------------------"
        HeLa_S3_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 400}
        HeLa_S3_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 221, 'max_bin': 155, 'min_child_samples': 180,
                                 'colsample_bytree': 0.7, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.2, 'learning_rate': 0.1, 'n_estimators': 200}
        HeLa_S3_dpcp_rf = {'n_estimators': 70, 'max_depth': 32, 'min_samples_leaf': 1, 'min_samples_split': 8,
                           'max_features': 'sqrt'}
        HeLa_S3_dpcp_svm = {'C': 32.0, 'gamma': 8.0, 'kernel': 'rbf'}
        HeLa_S3_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 3, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "----------------------------------------------"
        HeLa_S3_eiip_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 250}
        HeLa_S3_eiip_lightgbm = {'max_depth': -1, 'num_leaves': 281, 'max_bin': 5, 'min_child_samples': 110,
                                 'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.2, 'learning_rate': 0.1, 'n_estimators': 100}
        HeLa_S3_eiip_rf = {'n_estimators': 180, 'max_depth': 138, 'min_samples_leaf': 6, 'min_samples_split': 10,
                           'max_features': 'sqrt'}
        HeLa_S3_eiip_svm = {'C': 512.0, 'gamma': 64.0, 'kernel': 'rbf'}
        HeLa_S3_eiip_xgboost = {'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 3, 'gamma': 0,
                                'colsample_bytree': 0.6, 'subsample': 0.6, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "----------------------------------------------"
        HeLa_S3_kmer_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 250}
        HeLa_S3_kmer_lightgbm = {'max_depth': -1, 'num_leaves': 281, 'max_bin': 165, 'min_child_samples': 90,
                                 'colsample_bytree': 0.7, 'subsample': 0.9, 'subsample_freq': 70, 'reg_alpha': 0.001,
                                 'reg_lambda': 0.001, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 125}
        HeLa_S3_kmer_rf = {'n_estimators': 240, 'max_depth': 77, 'min_samples_leaf': 2, 'min_samples_split': 2,
                           'max_features': 'sqrt'}
        HeLa_S3_kmer_svm = {'C': 256.0, 'gamma': 16.0, 'kernel': 'rbf'}
        HeLa_S3_kmer_xgboost = {'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "----------------------------------------------"
        HeLa_S3_pseknc_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 300}
        HeLa_S3_pseknc_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 5, 'min_child_samples': 110,
                                   'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                   'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        HeLa_S3_pseknc_rf = {'n_estimators': 200, 'max_depth': 107, 'min_samples_leaf': 2, 'min_samples_split': 4,
                             'max_features': 'sqrt'}
        HeLa_S3_pseknc_svm = {'C': 1.0, 'gamma': 256.0, 'kernel': 'rbf'}
        HeLa_S3_pseknc_xgboost = {'n_estimators': 850, 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0,
                                  'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                  'learning_rate': 0.1}

        "=============================================="
        HUVEC_cksnap_deepforest = {"max_layers": 10, "n_estimators": 10, "n_trees": 300}
        HUVEC_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 271, 'max_bin': 45, 'min_child_samples': 10,
                                 'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 0.5,
                                 'reg_lambda': 0.5, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 175}
        HUVEC_cksnap_rf = {'n_estimators': 270, 'max_depth': 38, 'min_samples_leaf': 2, 'min_samples_split': 2,
                           'max_features': None}
        HUVEC_cksnap_svm = {'C': 128.0, 'gamma': 8.0, 'kernel': 'rbf'}
        HUVEC_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 12, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.6, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "----------------------------------------------"
        HUVEC_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 400}
        HUVEC_dpcp_lightgbm = {'max_depth': -1, 'num_leaves': 301, 'max_bin': 245, 'min_child_samples': 30,
                               'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 50, 'reg_alpha': 0.5,
                               'reg_lambda': 0.3, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 200}
        HUVEC_dpcp_rf = {'n_estimators': 300, 'max_depth': 61, 'min_samples_leaf': 2, 'min_samples_split': 3,
                         'max_features': 'log2'}
        HUVEC_dpcp_svm = {'C': 4.0, 'gamma': 16.0, 'kernel': 'rbf'}
        HUVEC_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 3, 'reg_lambda': 3,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        HUVEC_eiip_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 400}
        HUVEC_eiip_lightgbm = {'max_depth': -1, 'num_leaves': 281, 'max_bin': 25, 'min_child_samples': 80,
                               'colsample_bytree': 1.0, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                               'reg_lambda': 1e-05, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 250}
        HUVEC_eiip_rf = {'n_estimators': 310, 'max_depth': 28, 'min_samples_leaf': 1, 'min_samples_split': 2,
                         'max_features': 'sqrt'}
        HUVEC_eiip_svm = {'C': 16.0, 'gamma': 256.0, 'kernel': 'rbf'}
        HUVEC_eiip_xgboost = {'n_estimators': 600, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 0,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        HUVEC_kmer_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 400}
        HUVEC_kmer_lightgbm = {'max_depth': 0, 'num_leaves': 251, 'max_bin': 5, 'min_child_samples': 170,
                               'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 70, 'reg_alpha': 0.5,
                               'reg_lambda': 0.7, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 125}
        HUVEC_kmer_rf = {'n_estimators': 230, 'max_depth': 59, 'min_samples_leaf': 1, 'min_samples_split': 4,
                         'max_features': 'auto'}
        HUVEC_kmer_svm = {'C': 64.0, 'gamma': 16.0, 'kernel': 'rbf'}
        HUVEC_kmer_xgboost = {'n_estimators': 600, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 0,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        HUVEC_pseknc_deepforest = {'max_layers': 10, 'n_estimators': 2, 'n_trees': 250}
        HUVEC_pseknc_lightgbm = {'max_depth': 12, 'num_leaves': 261, 'max_bin': 235, 'min_child_samples': 110,
                                 'colsample_bytree': 0.9, 'subsample': 0.8, 'subsample_freq': 40, 'reg_alpha': 0.001,
                                 'reg_lambda': 0.001, 'min_split_gain': 1.0, 'learning_rate': 0.1, 'n_estimators': 225}
        HUVEC_pseknc_rf = {'n_estimators': 230, 'max_depth': 82, 'min_samples_leaf': 2, 'min_samples_split': 2,
                           'max_features': 'sqrt'}
        HUVEC_pseknc_svm = {'C': 0.5, 'gamma': 512.0, 'kernel': 'rbf'}
        HUVEC_pseknc_xgboost = {'n_estimators': 950, 'max_depth': 6, 'min_child_weight': 2, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "=============================================="
        IMR90_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 250}
        IMR90_cksnap_lightgbm = {'max_depth': 0, 'num_leaves': 271, 'max_bin': 95, 'min_child_samples': 60,
                                 'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                 'reg_lambda': 1e-05, 'min_split_gain': 0.3, 'learning_rate': 0.1, 'n_estimators': 225}
        IMR90_cksnap_rf = {'n_estimators': 280, 'max_depth': 124, 'min_samples_leaf': 1, 'min_samples_split': 2,
                           'max_features': 'auto'}
        IMR90_cksnap_svm = {'C': 16.0, 'gamma': 16.0, 'kernel': 'rbf'}
        IMR90_cksnap_xgboost = {'n_estimators': 900, 'max_depth': 10, 'min_child_weight': 2, 'gamma': 0.4,
                                'colsample_bytree': 0.6, 'subsample': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.1,
                                'learning_rate': 0.1}

        "----------------------------------------------"
        IMR90_dpcp_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        IMR90_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 281, 'max_bin': 115, 'min_child_samples': 20,
                               'colsample_bytree': 0.7, 'subsample': 1.0, 'subsample_freq': 50, 'reg_alpha': 0.0,
                               'reg_lambda': 0.0, 'min_split_gain': 0.5, 'learning_rate': 0.1, 'n_estimators': 125}
        IMR90_dpcp_rf = {'n_estimators': 70, 'max_depth': 116, 'min_samples_leaf': 1, 'min_samples_split': 9,
                         'max_features': 'log2'}
        IMR90_dpcp_svm = {'C': 1.0, 'gamma': 32.0, 'kernel': 'rbf'}
        IMR90_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 12, 'min_child_weight': 2, 'gamma': 0,
                              'colsample_bytree': 0.8, 'subsample': 0.6, 'reg_alpha': 0.05, 'reg_lambda': 0.1,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        IMR90_eiip_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        IMR90_eiip_lightgbm = {'max_depth': 13, 'num_leaves': 331, 'max_bin': 55, 'min_child_samples': 50,
                               'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 80, 'reg_alpha': 0.0,
                               'reg_lambda': 0.0, 'min_split_gain': 0.4, 'learning_rate': 0.2, 'n_estimators': 200}
        IMR90_eiip_rf = {'n_estimators': 240, 'max_depth': 78, 'min_samples_leaf': 1, 'min_samples_split': 2,
                         'max_features': 'auto'}
        IMR90_eiip_svm = {'C': 64.0, 'gamma': 128.0, 'kernel': 'rbf'}
        IMR90_eiip_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 1, 'gamma': 0,
                              'colsample_bytree': 0.6, 'subsample': 0.6, 'reg_alpha': 0, 'reg_lambda': 1,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        IMR90_kmer_deepforest = {'max_layers': 10, 'n_estimators': 5, 'n_trees': 400}
        IMR90_kmer_lightgbm = {'max_depth': 0, 'num_leaves': 271, 'max_bin': 175, 'min_child_samples': 120,
                               'colsample_bytree': 0.8, 'subsample': 1.0, 'subsample_freq': 30, 'reg_alpha': 0.7,
                               'reg_lambda': 0.9, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 200}
        IMR90_kmer_rf = {'n_estimators': 280, 'max_depth': 79, 'min_samples_leaf': 2, 'min_samples_split': 3,
                         'max_features': 'auto'}
        IMR90_kmer_svm = {'C': 128.0, 'gamma': 8.0, 'kernel': 'rbf'}
        IMR90_kmer_xgboost = {'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 2, 'gamma': 0.2,
                              'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                              'learning_rate': 0.1}

        "----------------------------------------------"
        IMR90_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 300}
        IMR90_pseknc_lightgbm = {'max_depth': -1, 'num_leaves': 321, 'max_bin': 45, 'min_child_samples': 30,
                                 'colsample_bytree': 0.9, 'subsample': 0.6, 'subsample_freq': 60, 'reg_alpha': 0.001,
                                 'reg_lambda': 0.0, 'min_split_gain': 0.9, 'learning_rate': 0.1, 'n_estimators': 100}
        IMR90_pseknc_rf = {'n_estimators': 280, 'max_depth': 28, 'min_samples_leaf': 4, 'min_samples_split': 7,
                           'max_features': 'auto'}
        IMR90_pseknc_svm = {'C': 512.0, 'gamma': 16.0, 'kernel': 'rbf'}
        IMR90_pseknc_xgboost = {'n_estimators': 950, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                                'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                                'learning_rate': 0.1}

        "=============================================="
        K562_cksnap_deepforest = {"max_layers": 20, "n_estimators": 2, "n_trees": 400}
        K562_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 311, 'max_bin': 225, 'min_child_samples': 60,
                                'colsample_bytree': 1.0, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                                'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.2, 'n_estimators': 250}
        K562_cksnap_rf = {'n_estimators': 330, 'max_depth': 109, 'min_samples_leaf': 2, 'min_samples_split': 3,
                          'max_features': 'sqrt'}
        K562_cksnap_svm = {'C': 16.0, 'gamma': 32.0, 'kernel': 'rbf'}
        K562_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 6, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 2, 'reg_lambda': 0.05,
                               'learning_rate': 0.1}

        "----------------------------------------------"
        K562_dpcp_deepforest = {"max_layers": 10, "n_estimators": 2, "n_trees": 300}
        K562_dpcp_lightgbm = {'colsample_bytree': 0.7, 'subsample': 0.7, 'subsample_freq': 80, 'reg_alpha': 1e-05,
                              'reg_lambda': 0.001, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 225}
        K562_dpcp_rf = {'n_estimators': 240, 'max_depth': 127, 'min_samples_leaf': 1, 'min_samples_split': 6,
                        'max_features': 'sqrt'}
        K562_dpcp_svm = {'C': 1.0, 'gamma': 32.0, 'kernel': 'rbf'}
        K562_dpcp_xgboost = {'n_estimators': 950, 'max_depth': 10, 'min_child_weight': 4, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 1, 'reg_lambda': 0.05,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        K562_eiip_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 300}
        K562_eiip_lightgbm = {'max_depth': 0, 'num_leaves': 321, 'max_bin': 225, 'min_child_samples': 110,
                              'colsample_bytree': 1.0, 'subsample': 0.7, 'subsample_freq': 0, 'reg_alpha': 1e-05,
                              'reg_lambda': 1e-05, 'min_split_gain': 0.1, 'learning_rate': 0.1, 'n_estimators': 150}
        K562_eiip_rf = {'n_estimators': 120, 'max_depth': 93, 'min_samples_leaf': 3, 'min_samples_split': 3,
                        'max_features': 'auto'}
        K562_eiip_svm = {'C': 32.0, 'gamma': 256.0, 'kernel': 'rbf'}
        K562_eiip_xgboost = {'n_estimators': 650, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        K562_kmer_deepforest = {'max_layers': 10, 'n_estimators': 8, 'n_trees': 300}
        K562_kmer_lightgbm = {'max_depth': 0, 'num_leaves': 321, 'max_bin': 5, 'min_child_samples': 70,
                              'colsample_bytree': 1.0, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.0,
                              'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 250}
        K562_kmer_rf = {'n_estimators': 290, 'max_depth': 137, 'min_samples_leaf': 10, 'min_samples_split': 7,
                        'max_features': None}
        K562_kmer_svm = {'C': 128.0, 'gamma': 16.0, 'kernel': 'rbf'}
        K562_kmer_xgboost = {'n_estimators': 650, 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        K562_pseknc_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 250}
        K562_pseknc_lightgbm = {'max_depth': 0, 'num_leaves': 221, 'max_bin': 55, 'min_child_samples': 70,
                                'colsample_bytree': 1.0, 'subsample': 1.0, 'subsample_freq': 70, 'reg_alpha': 1e-05,
                                'reg_lambda': 1e-05, 'min_split_gain': 0.1, 'learning_rate': 0.1, 'n_estimators': 100}
        K562_pseknc_rf = {'n_estimators': 320, 'max_depth': 63, 'min_samples_leaf': 6, 'min_samples_split': 2,
                          'max_features': 'sqrt'}
        K562_pseknc_svm = {'C': 0.5, 'gamma': 512.0, 'kernel': 'rbf'}
        K562_pseknc_xgboost = {'n_estimators': 1000, 'max_depth': 10, 'min_child_weight': 5, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                               'learning_rate': 0.1}

        "=============================================="
        NHEK_cksnap_deepforest = {"max_layers": 20, "n_estimators": 5, "n_trees": 400}
        NHEK_cksnap_lightgbm = {'max_depth': -1, 'num_leaves': 291, 'max_bin': 205, 'min_child_samples': 90,
                                'colsample_bytree': 1.0, 'subsample': 0.9, 'subsample_freq': 0, 'reg_alpha': 0.0,
                                'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 75}
        NHEK_cksnap_rf = {'n_estimators': 300, 'max_depth': 76, 'min_samples_leaf': 3, 'min_samples_split': 3,
                          'max_features': 'auto'}
        NHEK_cksnap_svm = {'C': 256.0, 'gamma': 8.0, 'kernel': 'rbf'}
        NHEK_cksnap_xgboost = {'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 2, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1,
                               'learning_rate': 0.1}

        "----------------------------------------------"
        NHEK_dpcp_deepforest = {"max_layers": 10, "n_estimators": 8, "n_trees": 200}
        NHEK_dpcp_lightgbm = {'max_depth': 0, 'num_leaves': 301, 'max_bin': 145, 'min_child_samples': 70,
                              'colsample_bytree': 0.7, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.9,
                              'reg_lambda': 1.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 150}
        NHEK_dpcp_rf = {'n_estimators': 300, 'max_depth': 138, 'min_samples_leaf': 1, 'min_samples_split': 5,
                        'max_features': 'auto'}
        NHEK_dpcp_svm = {'C': 8.0, 'gamma': 16.0, 'kernel': 'rbf'}
        NHEK_dpcp_xgboost = {'n_estimators': 1000, 'max_depth': 9, 'min_child_weight': 3, 'gamma': 0.5,
                             'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        NHEK_eiip_deepforest = {'max_layers': 10, 'n_estimators': 10, 'n_trees': 300}
        NHEK_eiip_lightgbm = {'max_depth': 11, 'num_leaves': 231, 'max_bin': 255, 'min_child_samples': 70,
                              'colsample_bytree': 1.0, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.0,
                              'reg_lambda': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 100}
        NHEK_eiip_rf = {'n_estimators': 230, 'max_depth': 56, 'min_samples_leaf': 2, 'min_samples_split': 6,
                        'max_features': 'log2'}
        NHEK_eiip_svm = {'C': 1024.0, 'gamma': 128.0, 'kernel': 'rbf'}
        NHEK_eiip_xgboost = {'n_estimators': 850, 'max_depth': 9, 'min_child_weight': 1, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 1, 'reg_lambda': 0.1,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        NHEK_kmer_deepforest = {'max_layers': 10, 'n_estimators': 10, 'n_trees': 300}
        NHEK_kmer_lightgbm = {'max_depth': 13, 'num_leaves': 261, 'max_bin': 115, 'min_child_samples': 60,
                              'colsample_bytree': 0.9, 'subsample': 0.9, 'subsample_freq': 40, 'reg_alpha': 0.0,
                              'reg_lambda': 0.001, 'min_split_gain': 1.0, 'learning_rate': 0.1, 'n_estimators': 150}
        NHEK_kmer_rf = {'n_estimators': 60, 'max_depth': 117, 'min_samples_leaf': 3, 'min_samples_split': 3,
                        'max_features': None}
        NHEK_kmer_svm = {'C': 16.0, 'gamma': 32.0, 'kernel': 'rbf'}
        NHEK_kmer_xgboost = {'n_estimators': 850, 'max_depth': 9, 'min_child_weight': 1, 'gamma': 0,
                             'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 1, 'reg_lambda': 0.1,
                             'learning_rate': 0.1}

        "----------------------------------------------"
        NHEK_pseknc_deepforest = {"max_layers": 20, "n_estimators": 13, "n_trees": 400}
        NHEK_pseknc_lightgbm = {'max_depth': 12, 'num_leaves': 251, 'max_bin': 105, 'min_child_samples': 80,
                                'colsample_bytree': 0.7, 'subsample': 0.6, 'subsample_freq': 0, 'reg_alpha': 0.7,
                                'reg_lambda': 0.7, 'min_split_gain': 0.0, 'learning_rate': 0.1, 'n_estimators': 125}
        NHEK_pseknc_rf = {'n_estimators': 40, 'max_depth': 62, 'min_samples_leaf': 6, 'min_samples_split': 8,
                          'max_features': 'log2'}
        NHEK_pseknc_svm = {'C': 2.0, 'gamma': 256.0, 'kernel': 'rbf'}
        NHEK_pseknc_xgboost = {'n_estimators': 650, 'max_depth': 8, 'min_child_weight': 5, 'gamma': 0,
                               'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.02,
                               'learning_rate': 0.1}


if __name__ == '_main_':
    print(getattr(EPIconst.ModelParams, "NHEK_pseknc_deepforest"))
