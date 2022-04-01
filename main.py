import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

from imputing import imputing
from categorical_feature_minipulation import categorical_feature_minipulation
from functions_from_stackEx import plot_grid_search

df = pd.read_csv('train.csv', index_col = 'Id')
df_test = pd.read_csv('test.csv', index_col = 'Id')
df_all = pd.concat([df, df_test])

df_all = imputing(df_all)
df_all = categorical_feature_minipulation(df_all)

df = df_all.iloc[:len(df)]
df_test = df_all.iloc[len(df):]

df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df = df.drop('SalePrice', axis = 1)
df_test = df_test.drop('SalePrice', axis = 1)

pd.set_option('display.max_rows', None)

print('\nTotal number of nans in df:',df.isna().sum().sum(),'\n',df.isna().sum())
print('\nTotal number of nans in df_test:',df_test.isna().sum().sum(),'\n',df_test.isna().sum(),'\n')

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
print('rf score', rf_score)
#rf score 0.14638716803443788

cv_results_split = rf_GS.cv_results_
plot_grid_search(cv_results_split, n_estimators, 'n_estimators')
plt.show()

#fit to all data so we can maybe improve the model more
rf_best_fit = rf_GS.fit(X, y)

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


print(df_pred)