
from LotFrontage_Imputer import LotFrontage_imputer
import numpy as np

def imputing(df):
	#changing nan to none for the relevant categorical features
	nan_is_none = ['MasVnrType','MiscFeature','Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageCond','GarageQual','Fence','PoolQC']
	df[nan_is_none] = df[nan_is_none].replace(np.nan,'None')

	#Setting numerical values associated with 'None' catigorical features to 0
	df.loc[(df['MasVnrType'] == 'None') , 'MasVnrArea' ] = 0
	df.loc[(df['GarageType'] == 'None') , 'GarageArea' ] = 0
	df.loc[(df['GarageType'] == 'None') , 'GarageCars' ] = 0
	df.loc[(df['GarageType'] == 'None') , 'GarageYrBlt' ] = 0

	# For electrical, mode, mean of non nan, and mean of nan are all dominate by the same value
	# So we set missing values to this

	df.loc[df['Electrical'].isna(), 'Electrical'] =  'SBrkr'

	#impute the missing values of lotFrontage
	#cannot place in pipeline as I dont know how to make my this method 
	#work in the pipeline framework :(
	#See imputing_exploration for detailed description

	lotFrontage_imputer = LotFrontage_imputer()
	lotFrontage_imputer.fit(df)
	df = lotFrontage_imputer.transform(df)

	#below we impute values that are missing in test.csv

	df['Utilities'] = df['Utilities'].fillna('AllPub')
	df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']] = df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].fillna(0)
	df[['BsmtHalfBath','BsmtFullBath']] = df[['BsmtHalfBath','BsmtFullBath']].fillna(0)

	modes = {}
	modes['Exterior1st'] = df['Exterior1st'].mode().values[0]
	modes['Exterior2nd'] = df['Exterior2nd'].mode().values[0]
	modes['SaleType'] = df['SaleType'].mode().values[0]
	modes['KitchenQual'] = df['KitchenQual'].mode().values[0]
	modes['Functional'] = df['Functional'].mode().values[0]

	df['Exterior1st'] = df['Exterior1st'].fillna(modes['Exterior1st'])
	df['Exterior2nd'] = df['Exterior2nd'].fillna(modes['Exterior2nd'])
	df['SaleType'] = df['SaleType'].fillna(modes['SaleType'])
	df['KitchenQual'] = df['KitchenQual'].fillna(modes['KitchenQual'])
	df['Functional'] = df['Functional'].fillna(modes['Functional'])

	df.loc[(df['GarageType'] == 'Detchd') & df['GarageCars'].isna()] = df.loc[
		(df['GarageType'] == 'Detchd') & df['GarageCars'].isna()
		].fillna(
			df.loc[df['GarageType'] == 'Detchd']['GarageCars'].mode()
			)

	#we use median as the data is skewed

	df.loc[(df['GarageType'] == 'Detchd') & df['GarageArea'].isna()] = df.loc[
		(df['GarageType'] == 'Detchd') & df['GarageArea'].isna()
		].fillna(
			df.loc[df['GarageType'] == 'Detchd']['GarageArea'].median()
			)

	df.loc[(df['GarageType'] == 'Detchd') & df['GarageYrBlt'].isna()] =df.loc[
		(df['GarageType'] == 'Detchd') & df['GarageYrBlt'].isna()
		].fillna(
			df.loc[df['GarageType'] == 'Detchd']['GarageYrBlt'].median()
			)

	#while replacing with RL in the cale 20 is clearly fine, the other two can probably get a better fit though more analysis
	df.loc[df['MSSubClass'] == 20, 'MSZoning'] = df.loc[df['MSSubClass'] == 20, 'MSZoning'].fillna('RL')
	df.loc[df['MSSubClass'] == 30, 'MSZoning'] = df.loc[df['MSSubClass'] == 30, 'MSZoning'].fillna('RM')
	df.loc[df['MSSubClass'] == 70, 'MSZoning'] = df.loc[df['MSSubClass'] == 70, 'MSZoning'].fillna('RM')

	return df


if __name__ == '__main__':
	import pandas as pd 
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.preprocessing import RobustScaler, FunctionTransformer
	from imputing import imputing

	df = pd.read_csv('train.csv', index_col = 'Id')
	df.drop('SalePrice', axis = 1)

	df_test = pd.read_csv('test.csv', index_col = 'Id')

	#df = pd.concat([df, df_test], ignore_index=True)

	df = imputing(df)
	df_test = imputing(df_test)

	print(df.info())
	print('')
	print(df_test.info())

	print('Total number of nans in df',df.isna().sum().sum())
	print('Total number of nans in df_test',df_test.isna().sum().sum())