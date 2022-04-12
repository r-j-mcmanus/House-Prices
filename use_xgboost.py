from xgboost import XGBRegressor
from xgboost import cv

from hyperopt import hp, fmin, Trials

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from functions_from_stackEx import plot_grid_search

import matplotlib.pyplot as plt
import numpy as np

# largly taken from 
# https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e

class use_xgboost:
	def __init__(self):
		# XGB parameters
		xgb_reg_params = {
			'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
			'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
			'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
			'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
			'subsample':        hp.uniform('subsample', 0.8, 1),
			'n_estimators':     hp.choice('n_estimators',    np.array([100, 200, 300])),
		}
		xgb_fit_params = {
			'eval_metric': 'rmse',
			'early_stopping_rounds': 10,
			'verbose': False
		}
		self.xgb_para = dict()
		self.xgb_para['reg_params'] = xgb_reg_params
		self.xgb_para['fit_params'] = xgb_fit_params

		self.xgb_best_fit = None


	def test_train(self, X_train, X_test, y_train, y_test):

		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

		trials = Trials()

		self.result = fmin(
			fn = self.fit_predict_score, 
			space = self.xgb_para, 
			max_evals = 200, 
			trials = trials
			)

		print('Best params')
		print(self.result)
		#best score: 

		self.xgb_best_fit = self.xgb_reg

		return self.result, trials


	def fit_predict_score(self, para):

		self.xgb_reg = XGBRegressor(**para['reg_params'])
		self.xgb_reg.fit(
				self.X_train, self.y_train
				, eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)]
				, **para['fit_params']
			)
		# Must have at least 1 validation dataset for early stopping.
		predictions = self.xgb_reg.predict(self.X_test)

		losses = MSE(self.y_test, predictions, squared=False)
		return losses