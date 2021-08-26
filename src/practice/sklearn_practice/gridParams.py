from sklearn.model_selection import ParameterGrid
"""
params
"""
parameters = [
    {
        'n_estimators' : [2, 3, 4, 5, 6, 7, 8, 9],
        'n_trees' : [x for x in range(50, 550, 50)],
        'predictors' : ['xgboost', 'lightgbm', 'forest'],
        'max_layers' : [layer for layer in range(20, 110, 10)],
        'use_predictor': [True]
    },
    {
        'n_estimators' : [2, 3, 4, 5, 6, 7, 8, 9],
        'n_trees' : [x for x in range(50, 550, 50)],
        'max_layers' : [layer for layer in range(20, 110, 10)]
    },

]


"""
gridSearchCV
"""
params_list = list(ParameterGrid(parameters))
