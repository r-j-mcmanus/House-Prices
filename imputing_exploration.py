import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv', index_col = 'Id')

print('We first replace nan with "None" for relevant columns')

df.loc[(df['MasVnrType'] == 'None') , 'MasVnrArea' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageArea' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageCars' ] = 0
df.loc[(df['GarageType'] == 'None') , 'GarageYrBlt' ] = 0

print('As we are interested in the accuracy in predicting the log of the sales price we add the log as a column and remove the sale price')
df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df = df.drop('SalePrice', axis = 1)

print('Note that area and length are fundamentally related, so we put them on simmilar units before any scaling. This will be usefull for linear regression.')
df['root_LotArea'] = df['LotArea'].apply(lambda x : x**0.5)

print('Due to the presence of many outliers, we use RobustScaler')
q_range_low = 5
q_range_high = 95
r_scaler_X = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (q_range_low,q_range_high), unit_variance = True)
r_scaler_y = RobustScaler(with_centering = True, with_scaling = True, quantile_range = (q_range_low,q_range_high), unit_variance = True)

print('inclusion of 1stFlrSF or GarageArea reduces the accuracy. using the root of the lotArea decreases the accuracy')
LotFrontage_corrs= ['LotFrontage','root_LotArea']  #  '1stFlrSF' 'GarageArea'

X = df[LotFrontage_corrs].dropna()

print('we scale the quantities, we do not need to prune the data yet as we are using a robust scaler, we can train it on all the data from the start.')
X_rescaled = r_scaler_X.fit_transform(X['root_LotArea'].values.reshape(-1, 1))
y_rescaled = r_scaler_y.fit_transform(X['LotFrontage'].values.reshape(-1, 1))

lower_quantile_value = np.quantile(X_rescaled, q_range_low/100)
upper_quantile_value = np.quantile(X_rescaled, q_range_high/100)

X_pruned = X_rescaled[X_rescaled > lower_quantile_value].reshape(-1, 1)
X_pruned = X_rescaled[X_rescaled < upper_quantile_value].reshape(-1, 1)
y_pruned = y_rescaled[X_rescaled > lower_quantile_value].reshape(-1, 1)
y_pruned = y_rescaled[X_rescaled < upper_quantile_value].reshape(-1, 1)

print('We recored the pruned values to see the effectivness of the linear regressor at extrapolation')
X_outlier_small = X_rescaled[X_rescaled < lower_quantile_value].reshape(-1, 1)
y_outlier_small = y_rescaled[X_rescaled < lower_quantile_value].reshape(-1, 1)
X_outlier_large = X_rescaled[X_rescaled > upper_quantile_value].reshape(-1, 1)
y_outlier_large = y_rescaled[X_rescaled > upper_quantile_value].reshape(-1, 1)


print('split the data to get an idea of accuracy')
X_train, X_test, y_train, y_test = train_test_split(X_pruned,y_pruned,random_state=42)

print('fit the linear regresser')
log_reg = LinearRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_small = log_reg.predict(X_outlier_small)
y_pred_large = log_reg.predict(X_outlier_large)

print("mean_squared_error on testing: {}".format(mean_squared_error(y_test, y_pred)))
print("mean_squared_error on small outliers: {}".format(mean_squared_error(y_outlier_small, y_pred_small)))
print("mean_squared_error on large outliers: {}".format(mean_squared_error(y_outlier_large, y_pred_large)))

print('We see that the model fits the bulk and the small outliers well')
print('it does not do well on the large outliers')

print('We now fit to the whole pruned data and use it to impute the missing lot front size.')
log_reg.fit(X_pruned, y_pruned)

print('we get the values of root_LotArea for missing values of LotFrontage')
X_nan = df[ df['LotFrontage'].isna() ]['root_LotArea']
print('we scale the values and predict the values for the scaled LotFrontage')
X_nan_rescaled = r_scaler_X.transform(X_nan.values.reshape(-1, 1))
y_nan_rescaled = log_reg.predict(X_nan_rescaled)

plt.plot(X_rescaled,y_rescaled, linestyle = 'None', color = 'blue', marker = 'x', alpha = 0.8)
plt.plot(X_nan_rescaled,y_nan_rescaled, linestyle = 'None', color = 'red', marker = 'x', alpha = 0.8)

plt.plot(
	X_rescaled[X_rescaled > upper_quantile_value],
	y_rescaled[X_rescaled > upper_quantile_value], 
	linestyle = 'None', color = 'yellow', marker = 'x', alpha = 1)

print('we see that the values that are mising within the lowwer 95% percentile seem rasonable')
print('but values in the upper 5% do not seem to have much justification. they all lie will above the mean and median:')
print('upper 5% mean: ', y_outlier_large.mean())
print('upper 5% median: ', np.median(y_outlier_large))
print('upper 5% variance: ', np.var(y_outlier_large))
print('it seems the two variables are not linearly correlated over the whole range.')

y_upper_percentile_prediction = log_reg.predict([[upper_quantile_value]])

print('mean_squared_error from mean: ', ((y_outlier_large-y_outlier_large.mean())**2).sum() / len(y_outlier_large) )
print('mean_squared_error from median: ', ((y_outlier_large-np.median(y_outlier_large))**2).sum() / len(y_outlier_large) )

print('regresser value at 95%:', y_upper_percentile_prediction)
print('mean_squared_error from regersor value at 95%:', ((y_outlier_large-y_upper_percentile_prediction)**2).sum() / len(y_outlier_large) )

print('That the mean has a better RMS in this region means we shouldnt trust the regresser')
print('We could split the data into bins and train on each bin')
print('However, we will just use the mean')

y_nan_rescaled_with_mean = [ log_reg.predict([x]) if x < upper_quantile_value else y_outlier_large.mean() for x in X_nan_rescaled]

plt.plot(X_nan_rescaled,y_nan_rescaled_with_mean, linestyle = 'None', color = 'green', marker = 'x', alpha = 0.8)


plt.show()


from imputing import imputing

df_test = pd.read_csv('test.csv', index_col = 'Id')
df_test = imputing(df_test)

print(df_test.info())
has_nan = ['MSZoning']

print(df_test.info())

print(df_test['MSZoning'].value_counts())
print(df_test[df_test['MSZoning'].isna()]['MSSubClass'])
print(20, df_test[df_test['MSSubClass'] == 20]['MSZoning'].value_counts())
print(30, df_test[df_test['MSSubClass'] == 30]['MSZoning'].value_counts())
print(70, df_test[df_test['MSSubClass'] == 70]['MSZoning'].value_counts())