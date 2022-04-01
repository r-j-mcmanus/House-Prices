import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer

def categorical_feature_minipulation(df):
	pass





if __name__ == '__main__':
	from imputing import imputing

	pd.set_option('display.max_columns', None)
	df = pd.read_csv('train.csv', index_col = 'Id')
	
	imputer = LotFrontage_imputer()
	imputer.fit(df)
	print(df[['LotFrontage','LotArea']].head(10))
	df = imputer.transform(df)