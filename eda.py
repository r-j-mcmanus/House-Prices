import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv', index_col = 'Id')

df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df = df.drop('SalePrice', axis = 1)

print(df.head())
print(df.info())
#print(df.describe())

columns_with_null = []
numbers_of_null = []
numberic_with_null = []
non_numeric_with_null = []
colours_for_plot = []

for col in df.columns:
	number_of_null = df[col].isnull().sum()
	if df[col].isnull().sum() != 0:
		columns_with_null.append(col)
		numbers_of_null.append(number_of_null)
		if df[col].dtype != object:
			numberic_with_null.append(col)
			colours_for_plot.append('red')
		else:
			non_numeric_with_null.append(col)
			colours_for_plot.append('blue')

print('We see that there are a lot of collomns with null values.')
print('Of these, ', numberic_with_null,' are numeric.')
print('GarageYrBlt is associated with the presence of a Garage, which is why GarageType and GarageFinish are missing in the same ammount')
print('while LotFrontage and MasVnrArea are values that can be interpolated.\n')

print('To study LotFrontage and MasVnrArea, we examine their corillations\n')

corrM = df.corr()

print(corrM[corrM['LotFrontage'] > 0.3]['LotFrontage'],'\n')
print(corrM[corrM['MasVnrArea'] > 0.3]['MasVnrArea'])

print('\nWe see that both are correlated with several other quantities.')
print('Hence we can predict the missing values from a predictive model using these correlations.')

print('We could also drop the entries that are missing MasVnrArea\n\n')

fig1 = plt.figure()
plt.bar(columns_with_null,numbers_of_null,color = colours_for_plot)
plt.xticks(rotation=90)
plt.title('Number of NaNs')
plt.tight_layout()
plt.show()

print('For the non numeric data, we would like to know the distribution of values')
print('In particular, it would be nice to just use the median values, hence we find the collumns which have a dominating value.')

percent_mean = {}

for col in non_numeric_with_null:
	df_col_val_counts = df[col].transpose().value_counts()
	df_col_val_counts.plot.bar()
	plt.title(df_col_val_counts.name)
	#plt.show()
	percent_mean[col] = df_col_val_counts.iloc[0] / df_col_val_counts.sum()

percent_mean = pd.Series(percent_mean)
percent_mean_large = percent_mean[percent_mean > 0.9]
percent_mean_small = percent_mean[percent_mean < 0.9]
print(percent_mean_large)

print('We see that ',percent_mean_large.index.values,' all have a substantial percentage of values that are equal to the mode. In cases where we expect a value, we can use the mode.\n')


print(percent_mean_small)
print('However, ', percent_mean_small.index.values,'all have a small value and so we should put in more work.\n\n')


print('Next we are concerned with how skew the data is. Skew data can lead to problems in various ML algroithems, and so the data should be sclaed and normalised approprietly. ')

print(df.skew(axis = 0, numeric_only=True))


print('\nNote that we find a few values with very large skew.')

df_object = df.select_dtypes(include = 'object')
df_float64 = df.select_dtypes(include = 'float64')
df_int64 = df.select_dtypes(include = 'int64')


df_float64.hist(bins = 30)
df_int64.hist(bins = 30)

print('It is not hard to see that some of the distributions have outliers that offset this ')

plt.show()

