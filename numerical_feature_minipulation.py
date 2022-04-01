import pandas as pd 
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler

def numerical_feature_minipulation(df):
	#df = using_r_scaler(df, 'SalePrice')
	df['TotalBuildingSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

	df = using_r_scaler(df, 'LotFrontage')
	df = using_r_scaler(df, 'LotArea')
	df = using_MinMaxScaler(df, 'YearBuilt')
	df = using_MinMaxScaler(df, 'YearRemodAdd')
	df = using_r_scaler(df, 'MasVnrArea', Lambda = 0.1, with_centering = False)
	df = using_r_scaler(df, 'BsmtFinSF1', Lambda = 0.1)
	df = using_r_scaler(df, 'BsmtFinSF2', Lambda = 0.1)
	df = using_r_scaler(df, 'BsmtUnfSF', Lambda = 0.1)
	df = using_r_scaler(df, 'TotalBsmtSF', Lambda = 0.1)
	df = using_r_scaler(df, '1stFlrSF')
	df = using_r_scaler(df, '2ndFlrSF', Lambda = 0.1)
	df = using_r_scaler(df, 'LowQualFinSF', Lambda = 0.1)
	df = using_r_scaler(df, 'GrLivArea')
	df = using_MinMaxScaler(df, 'GarageYrBlt')
	df = using_r_scaler(df, 'GarageArea', Lambda = 0.1)
	df = using_r_scaler(df, 'WoodDeckSF', Lambda = 0.1)
	df = using_r_scaler(df, 'OpenPorchSF', Lambda = 0.1)
	df = using_r_scaler(df, 'EnclosedPorch', Lambda = 0.1)
	df = using_r_scaler(df, '3SsnPorch', Lambda = 0.1)
	df = using_r_scaler(df, 'ScreenPorch', Lambda = 0.1)
	df = using_r_scaler(df, 'PoolArea', Lambda = 0.1)
	df = using_r_scaler(df, 'MiscVal', Lambda = 0.1)
	df = using_r_scaler(df, 'TotalBuildingSF')
	df = using_MinMaxScaler(df, 'YrSold')
	return df

def using_r_scaler(df, name, Lambda = 0,  lower_quantile_range = 0.05, upper_quantile_range = 0.95, with_centering = True, unit_variance = True):
	r_scaler = RobustScaler(with_centering = with_centering, with_scaling = True, quantile_range = (lower_quantile_range,upper_quantile_range), unit_variance = unit_variance)
	
	if Lambda == 0:
		df[name] = df[name].apply(lambda x : np.log(x))
		df[name] = r_scaler.fit_transform(df[name].values.reshape(-1, 1))

	else:
		#fn from p 32 of applied predictive modeling
		df[name] = df[name].apply(lambda x : (x ** Lambda - 1) / Lambda )
		df[name] = r_scaler.fit_transform(df[name].values.reshape(-1, 1))

	return df

def using_MinMaxScaler(df, name):
	mm_scaler = MinMaxScaler()
	df[name] = mm_scaler.fit_transform(df[name].values.reshape(-1, 1))

	return df


def plot_using_r_scaler(df, name, Lambda = 0,  lower_quantile_range = 0.05, upper_quantile_range = 0.95, with_centering = True, unit_variance = True):
	r_scaler = RobustScaler(with_centering = with_centering, with_scaling = True, quantile_range = (lower_quantile_range,upper_quantile_range), unit_variance = unit_variance)
	
	if Lambda == 0:
		df['log_'+name] = df[name].apply(lambda x : np.log(x))
		df['log_'+name+'_rescaled'] = r_scaler.fit_transform(df['log_'+name].values.reshape(-1, 1))
	
		df[[name,'log_'+name,'log_'+name+'_rescaled']].hist(bins = 20)

	else:
		#fn from p 32 of applied predictive modeling
		df['lambda_'+str(Lambda)+'_'+name] = df[name].apply(lambda x : (x ** Lambda - 1) / Lambda )
		df['lambda_'+str(Lambda)+'_'+name+'_rescaled'] = r_scaler.fit_transform(df['lambda_'+str(Lambda)+'_'+name].values.reshape(-1, 1))

		df[[name,'lambda_'+str(Lambda)+'_'+name,'lambda_'+str(Lambda)+'_'+name+'_rescaled']].hist(bins = 20)

	plt.show()
	return df

def plot_using_MinMaxScaler(df, name):
	mm_scaler = MinMaxScaler()
	df[name+'_rescaled'] = mm_scaler.fit_transform(df[name].values.reshape(-1, 1))

	df[[name, name+'_rescaled']].hist(bins = 20)
	plt.show()
	return df

if __name__ == '__main__':
	from imputing import imputing
	from scipy.stats import norm
	import matplotlib.pyplot as plt
	import matplotlib.mlab as mlab

	from sklearn.preprocessing import RobustScaler


	pd.set_option('display.max_columns', None)
	df = pd.read_csv('train.csv', index_col = 'Id')

	print(df.info())

	df = imputing(df)
	print(df[['LotFrontage','LotArea']].head(10))

	################ SalePrice ###################

	df = plot_using_r_scaler(df, 'SalePrice')
	df = plot_using_r_scaler(df, 'LotFrontage')
	df = plot_using_r_scaler(df, 'LotArea')
	df = plot_using_MinMaxScaler(df, 'YearBuilt')
	df = plot_using_MinMaxScaler(df, 'YearRemodAdd')
	df = plot_using_r_scaler(df, 'MasVnrArea', Lambda = 0.1, with_centering = False)
	df = plot_using_r_scaler(df, 'BsmtFinSF1', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'BsmtFinSF2', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'BsmtUnfSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'TotalBsmtSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, '1stFlrSF')
	df = plot_using_r_scaler(df, '2ndFlrSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'LowQualFinSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'GrLivArea')
	df = plot_using_MinMaxScaler(df, 'GarageYrBlt')
	df = plot_using_r_scaler(df, 'GarageArea', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'WoodDeckSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'OpenPorchSF', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'EnclosedPorch', Lambda = 0.1)
	df = plot_using_r_scaler(df, '3SsnPorch', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'ScreenPorch', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'PoolArea', Lambda = 0.1)
	df = plot_using_r_scaler(df, 'MiscVal', Lambda = 0.1)
	df = plot_using_MinMaxScaler(df, 'YrSold')
