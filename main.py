import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

from imputing import imputing
from categorical_feature_minipulation import categorical_feature_minipulation
from numerical_feature_minipulation import numerical_feature_minipulation
from functions_from_stackEx import plot_grid_search
from remove_features import remove_features_labels

print('reading files\n')
df = pd.read_csv('train.csv', index_col = 'Id')
df_test = pd.read_csv('test.csv', index_col = 'Id')
df_all = pd.concat([df, df_test])

print('imputing\n')
df_all = imputing(df_all)
print('categorical_feature_minipulation\n')
df_all = categorical_feature_minipulation(df_all)
print('numerical_feature_minipulation\n')
df_all = numerical_feature_minipulation(df_all)
print('remove_features_labels\n')
col_to_drop = remove_features_labels(df_all.iloc[:len(df)], cutoff = 10 ** -4)
df_all = df_all.drop(col_to_drop, axis = 1)

print('Splitting the imported data back into test and train')
df = df_all.iloc[:len(df)]
df_test = df_all.iloc[len(df):]

df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df_salePrice = df['SalePrice']
df = df.drop('SalePrice', axis = 1)
df_test = df_test.drop('SalePrice', axis = 1)

pd.set_option('display.max_rows', None)

print('\nTotal number of nans in df_all:',df_all.isna().sum().sum(),'\n',df_all.isna().sum())
print('\nNumber of collomns:',len(df_test.columns),'\n')

pd.set_option('display.max_rows', 10)

#split the features from the 
y = df['log_SalePrice'].values
X = df.drop('log_SalePrice', axis = 1).values

#make out training set to ensure our hyperparams arnt over fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print('RMS using the mean for a worst case scenario')
mean_score = (((y_test - y_train.mean())**2).sum() / len(y_test))**(1/2)
print('mean score', mean_score)
#mean score 0.43324686081152136

n_estimators = [ 700, 770, 850, 930, 1000, 1100, 1300]
param_grid = {
	'n_estimators' : n_estimators
	}
rf = RandomForestRegressor(random_state = 42)
rf_GS = GridSearchCV(rf, param_grid = param_grid, cv = 5, n_jobs = -1, scoring = 'neg_root_mean_squared_error')
rf_best_fit = rf_GS.fit(X_train, y_train)

y_pred = rf_best_fit.predict(X_test)
rf_score = MSE(y_test, y_pred, squared=False)
print('test rf score', rf_score)
#rf score 0.14470006197595744

y_pred_train = rf_best_fit.predict(X_train)
rf_score = MSE(y_train, y_pred_train, squared=False)
print('train!!! rf score', rf_score)

cv_results_split = rf_GS.cv_results_
plot_grid_search(cv_results_split, n_estimators, 'n_estimators')
plt.show()

plt.plot(y_test, y_pred, linestyle = 'None', marker='x')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

plt.plot(y_test, (y_pred-y_test), linestyle = 'None', marker='x')
plt.xlabel('y_test')
plt.ylabel('y_pred-y_test')
plt.show()

residual_test = (y_pred-y_test)
prediction_var_test = residual_test.var()
outliers_test = [ y_test[residual_test ** 2 > 3 * prediction_var_test] , y_pred[residual_test ** 2 > 3 * prediction_var_test] ] 
inliers_test = [  y_test[residual_test ** 2 < 3 * prediction_var_test] , y_pred[residual_test ** 2 < 3 * prediction_var_test] ]
plt.plot(*outliers_test, linestyle = 'None', marker='x', color = 'red')
plt.plot(*inliers_test, linestyle = 'None', marker='x', color = 'blue')

residual_train = (y_pred_train-y_train)
prediction_var_train = residual_train.var()
outliers_train = [ y_train[residual_train ** 2 > 3 * prediction_var_train] , y_pred_train[residual_train ** 2 > 3 * prediction_var_train] ] 
inliers_train = [  y_train[residual_train ** 2 < 3 * prediction_var_train] , y_pred_train[residual_train ** 2 < 3 * prediction_var_train] ]
plt.plot(*outliers_train, linestyle = 'None', marker='o', color = 'red', alpha = 0.4)
plt.plot(*inliers_train, linestyle = 'None', marker='o', color = 'blue', alpha = 0.4)
plt.xlabel('y_train')
plt.ylabel('y_pred_train')
plt.title('outliers')
plt.show()



#fit to all data so we can maybe improve the model more
rf_best_fit = rf_GS.fit(X, y)

y_pred = rf_best_fit.predict(X)

residual = (y_pred-y)
prediction_var = residual.var()
outliers = [ y[residual ** 2 > 3 * prediction_var] , y_pred[residual ** 2 > 3 * prediction_var] ] 
inliers = [  y[residual ** 2 < 3 * prediction_var] , y_pred[residual ** 2 < 3 * prediction_var] ]
plt.plot(*outliers, linestyle = 'None', marker='o', color = 'red', alpha = 0.4)
plt.plot(*inliers, linestyle = 'None', marker='o', color = 'blue', alpha = 0.4)
plt.xlabel('y')
plt.ylabel('y_pred')
plt.title('outliers - all ')
plt.show()

################ Applying to test dataset ####################

X_test = df_test.values
y_pred = rf_best_fit.predict(X_test)

y_pred = np.exp(y_pred)

df_pred = pd.DataFrame(
	y_pred
	,index =df_test.index 
	,columns=['SalePrice'] 
	)

df_pred.to_csv('SalePrice_predictions.csv')


df_salePrice.hist()
df_pred.hist('SalePrice')
