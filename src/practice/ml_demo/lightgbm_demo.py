import lightgbm as lgb
import time

params = {'max_bin': 63,
          'num_leaves': 255,
          'learning_rate': 0.1,
          'tree_learner': 'serial',
          'task': 'train',
          'is_training_metric': 'false',
          'min_data_in_leaf': 1,
          'min_sum_hessian_in_leaf': 100,
          'ndcg_eval_at': [1, 3, 5, 10],
          'sparse_threshold': 1.0,
          'device': 'gpu',
          'gpu_platform_id': 0,
          'gpu_device_id': 0}

dtrain = lgb.Dataset('data/higgs.train')
t0 = time.time()
gbm = lgb.train(params, train_set=dtrain, num_boost_round=10,
                valid_sets=None, valid_names=None,
                fobj=None, feval=None, init_model=None,
                feature_name='auto', categorical_feature='auto',
                early_stopping_rounds=None, evals_result=None,
                verbose_eval=True,
                keep_training_booster=False, callbacks=None)
t1 = time.time()

print('gpu version elapse time: {}'.format(t1 - t0))

params = {'max_bin': 63,
          'num_leaves': 255,
          'learning_rate': 0.1,
          'tree_learner': 'serial',
          'task': 'train',
          'is_training_metric': 'false',
          'min_data_in_leaf': 1,
          'min_sum_hessian_in_leaf': 100,
          'ndcg_eval_at': [1, 3, 5, 10],
          'sparse_threshold': 1.0,
          'device': 'cpu'
          }

t0 = time.time()
gbm = lgb.train(params, train_set=dtrain, num_boost_round=10,
                valid_sets=None, valid_names=None,
                fobj=None, feval=None, init_model=None,
                feature_name='auto', categorical_feature='auto',
                early_stopping_rounds=None, evals_result=None,
                verbose_eval=True,
                keep_training_booster=False, callbacks=None)
t1 = time.time()

print('cpu version elapse time: {}'.format(t1 - t0))