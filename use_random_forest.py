from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from functions_from_stackEx import plot_grid_search
import matplotlib.pyplot as plt


class use_random_forest:
	def __init__(self):
		self.n_estimators = [ 700, 770, 850, 930, 1000, 1100, 1300]
		self.param_grid = {
			'n_estimators' : self.n_estimators
		}
		self.rf = RandomForestRegressor(random_state = 42)
		self.rf_GS = GridSearchCV(self.rf, param_grid = self.param_grid, cv = 5, n_jobs = -1, scoring = 'neg_root_mean_squared_error')

		self.rf_best_fit = None


	def test_train(self, X_train, X_test, y_train, y_test):
		
		self.rf_best_fit = self.rf_GS.fit(X_train, y_train)

		y_pred = self.rf_best_fit.predict(X_test)
		rf_score = MSE(y_test, y_pred, squared=False)
		print('test rf score', rf_score)
		#rf score 0.14470006197595744

		y_pred_train = self.rf_best_fit.predict(X_train)
		rf_score = MSE(y_train, y_pred_train, squared=False)
		print('train!!! rf score', rf_score)

		cv_results_split = self.rf_GS.cv_results_
		plot_grid_search(cv_results_split, n_estimators, 'n_estimators')
		plt.show()

		return y_pred

	def all(self, X, y):
		self.rf_best_fit = self.rf_GS.fit(X, y)
		y_pred = self.rf_best_fit.predict(X)
		return y_pred