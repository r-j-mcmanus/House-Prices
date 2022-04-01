import pandas as pd 
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def remove_features_labels(df, cutoff = 10 ** -4):
	mutial_info = mutual_info_regression(df.drop('SalePrice', axis = 1), df['SalePrice'])
	df_info = pd.DataFrame(mutial_info, index = df.drop('SalePrice', axis = 1).columns, columns = ['info'])

	col_to_drop = df_info[df_info['info'] < cutoff].index

	return col_to_drop


if __name__ == '__main__':

	from imputing import imputing
	from categorical_feature_minipulation import categorical_feature_minipulation
	from numerical_feature_minipulation import numerical_feature_minipulation


	df = pd.read_csv('train.csv', index_col = 'Id')
	df_test = pd.read_csv('test.csv', index_col = 'Id')
	df_all = pd.concat([df, df_test])

	df_all = imputing(df_all)
	df_all = categorical_feature_minipulation(df_all)
	df_all = numerical_feature_minipulation(df_all)
	
	df = df_all.iloc[:len(df)]

	mutial_info = mutual_info_regression(df.drop('SalePrice', axis = 1), df['SalePrice'])
	df_info = pd.DataFrame(mutial_info, index = df.drop('SalePrice', axis = 1).columns, columns = ['info'])

	col_to_drop = df_info[df_info['info'] < 10 ** -4].index

	print(col_to_drop)
