import pandas as pd 
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

''''
I would like to get this working with pipeline, but sunken cost.

See bellow for start
'''

class LotFrontage_imputer:
	def __init__(self, q_range_high=0.95):
		self.q_range_high = q_range_high
		self.r_scaler_X = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (0.01,q_range_high), unit_variance = True)
		self.r_scaler_y = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (0.01,q_range_high), unit_variance = True)
		self.y_upper_quantile_mean = 0
		self.lin_reg = LinearRegression()

	def fit(self,df):
		LotFrontage_corrs= ['LotFrontage','LotArea'] 
		
		X = df[LotFrontage_corrs]
		X = X.dropna(axis = 0)
		X['LotArea'] = X['LotArea'].apply(lambda x : x**0.5)
		
		X_rescaled = self.r_scaler_X.fit_transform(X['LotArea'].values.reshape(-1, 1))
		y_rescaled = self.r_scaler_y.fit_transform(X['LotFrontage'].values.reshape(-1, 1))

		self.upper_quantile_value = np.quantile(X_rescaled, self.q_range_high)
		self.y_upper_quantile_mean = y_rescaled[X_rescaled > self.upper_quantile_value].mean()
		
		X_pruned = X_rescaled[X_rescaled < self.upper_quantile_value].reshape(-1, 1)
		y_pruned = y_rescaled[X_rescaled < self.upper_quantile_value].reshape(-1, 1)

		self.lin_reg.fit(X_pruned, y_pruned)
		return self

	def transform(self,df):
		LotFrontage_corrs= ['LotFrontage','LotArea']

		X = df[ df['LotFrontage'].isna() ]['LotArea']
		X = X.apply(lambda x : x**0.5)
		X = X.values.reshape(-1, 1)
		X = self.r_scaler_X.transform(X)

		y_pred_scaled = np.array([ self.lin_reg.predict([x]) if x < self.upper_quantile_value else self.y_upper_quantile_mean for x in X])
		y_pred_scaled = y_pred_scaled.reshape(-1, 1)
		y_pred = self.r_scaler_y.inverse_transform(y_pred_scaled)
		df.loc[df['LotFrontage'].isna() ,'LotFrontage'] = y_pred
		return df


if __name__ == '__main__':
	pd.set_option('display.max_columns', None)
	df = pd.read_csv('train.csv', index_col = 'Id')
	imputer = LotFrontage_imputer()
	imputer.fit(df)
	print(df[['LotFrontage','LotArea']].head(10))
	df = imputer.transform(df)
	print(df[['LotFrontage','LotArea']].head(10))

#####################


class impute_lower_quartile_with_linreg():
	'''Replace nan in array y with the a linear regresion from feature X up to a given percentile. 
	For value of X greater than the percentile, the mean of all values greater than that percentile is used. 
    Attributes
    ----------
    '''
	def __init__(self, q_range_high = 95):
		self.q_range_high = q_range_high
		self.r_scaler_X = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (0,q_range_high), unit_variance = True)
		self.r_scaler_y = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (0,q_range_high), unit_variance = True)
		self.upper_quantile_value = 0
		self.upper_quantile_mean = 0
		self.lig_reg = LinearRegression()

		return self

	def fit(self, X, y):
		X = check_array(X)
		y = check_array(y)
		X = X ** 0.5

		X_rescaled = r_scaler_X.fit_transform(X)
		y_rescaled = r_scaler_y.fit_transform(y)

		self.upper_quantile_value = np.quantile(X_rescaled, q_range_high/100)
		self.upper_quantile_mean = y_rescaled[X_rescaled > self.upper_quantile_value].mean()

		X_pruned = X_rescaled[X_rescaled < self.upper_quantile_value]
		y_pruned = y_rescaled[X_rescaled < self.upper_quantile_value]
		
		self.lig_reg.fit(X_pruned, y_pruned)

		return self

	def transform(self, X, y):
		X = check_array(X)
		y = check_array(y)
		assert(len(X) == len(y))

		#return [ y if self.lig_reg.predict(x) if x < self.upper_quantile_value else upper_quantile_mean for x in X ]
