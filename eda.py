import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv', index_col = 'Id')

df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df = df.drop('SalePrice', axis = 1)

print('We start by chaning columns where nan == "none" ')
nan_is_none = ['MasVnrType','MiscFeature','Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageCond','GarageQual','Fence','PoolQC']
print('These columns are ', nan_is_none,'\n')
df[nan_is_none] = df[nan_is_none].replace(np.nan,'None')

#print(df.head())
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

print('\nWe see that there are a lot of collomns with null values.')
print('Of these, ', numberic_with_null,' are numeric.')
print('LotFrontage and MasVnrArea are values that can be interpolated.')
print('missing GarageYrBlt is an int but should be "none" as they corrispond to no Garage. We will put to zero\n')

print('To study LotFrontage and MasVnrArea, we examine their corillations\n')

corrM = df.corr()

LotFrontage_corrs = corrM[corrM['LotFrontage'] > 0.3]['LotFrontage'].drop('LotFrontage')
MasVnrArea_corrs = corrM[corrM['MasVnrArea'] > 0.3]['MasVnrArea'].drop('MasVnrArea')

print(LotFrontage_corrs,'\n')
print(MasVnrArea_corrs)

fig, axes = plt.subplots(1,len(LotFrontage_corrs.index), sharey=True)

for i, col in enumerate(LotFrontage_corrs.index):
    df.plot(x=[col], y=["LotFrontage"], kind="scatter", ax=axes[i], logx=True, logy=True)

plt.show()
#plot could be better


print('\nWe see that both are correlated with several other quantities.')
print('Hence we can predict the missing values from a predictive model using these correlations.')
print('We could also just drop the entries that are missing MasVnrArea\n\n')

df['GarageYrBlt'].hist()

df.loc[(df['MasVnrType'] == 'None') , 'MasVnrArea' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageArea' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageCars' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageYrBlt' ] = 0

df['GarageYrBlt'].hist()
plt.show()

print('\n') 

fig1 = plt.figure()
plt.bar(columns_with_null,numbers_of_null,color = colours_for_plot)
plt.xticks(rotation=90)
plt.title('Number of NaNs')
plt.tight_layout()
plt.show()

print('For the non numeric data, we would like to know the distribution of values')
print('In particular, it would be nice to just use the median values, hence we find the collumns which have a dominating value.')

percent_mode = {}

for col in non_numeric_with_null:
	df_col_val_counts = df[col].transpose().value_counts()
	df_col_val_counts.plot.bar()
	plt.title(df_col_val_counts.name)
	#plt.show()
	percent_mode[col] = [df_col_val_counts[0] / df_col_val_counts.sum(), df_col_val_counts[0] / 1460, df_col_val_counts.sum() / 1460]


percent_mode = pd.DataFrame(percent_mode, index = ['mean_sum','mean_total','non_nan']).transpose()
percent_mode_large = percent_mode[percent_mode['mean_sum'] > 0.9]
percent_mode_small = percent_mode[percent_mode['mean_sum'] < 0.9]

print(percent_mode_large)

print('\nWe see that ',percent_mode_large.index.values,' has a substantial percentage of values that are equal to the mode. In cases where we expect a value, we can use the mode.\n')


print(percent_mode_small)
print('However, ', percent_mode_small.index.values,'all have a small value and so we are done.\n\n')

print('Next we are concerned with how skew the data is. Skew data can lead to problems in various ML algroithems, and so the data should be sclaed and normalised approprietly. ')

df_skew = df.skew(axis = 0, numeric_only=True)
print(df_skew)

print('\nNote that we find a few values with very large skew.')
print('It is not hard to see that some of the distributions have outliers that offset the skew ')

df_no_object = df.select_dtypes(exclude = 'object')
df_object = df.select_dtypes(include = 'object')
df_float64 = df.select_dtypes(include = 'float64')
df_int64 = df.select_dtypes(include = 'int64')

i=0;
for col in df_float64:
	df_float64 = df_float64[df_float64[col] > df_float64[col].quantile(0.025)]
	df_float64 = df_float64[df_float64[col] < df_float64[col].quantile(0.975)]
	if i == 100:
		break

print('To fix this we look at the inner 95\% of points')
print(df_float64.skew(axis = 0, numeric_only=True))

df_float64.hist(bins = 20)
df_int64.hist(bins = 20)
plt.show()

print('It makes no sence to try and map GarageYrBlt to a gaussian, while MasVnrArea, LotFrontage and log_SalePrice could be minipulated.')