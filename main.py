import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

from imputing import imputing
from categorical_feature_minipulation import categorical_feature_minipulation
from numerical_feature_minipulation import numerical_feature_minipulation
from remove_features import remove_features_labels

from use_random_forest import use_random_forest
from use_xgboost import use_xgboost

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print('RMS using the mean for a worst case scenario')
mean_score = (((y_test - y_train.mean())**2).sum() / len(y_test))**(1/2)
print('mean score', mean_score)
#mean score 0.43324686081152136

model_maker = use_xgboost()
xgb_model = model_maker.test_train(X_train, X_test, y_train, y_test)
y_pred = model_maker.xgb_best_fit.predict(X_test)

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
plt.show()

#fit to all data so we can maybe improve the model more
#y_pred = model.all(X, y)

#residual = (y_pred-y)
#prediction_var = residual.var()
#outliers = [ y[residual ** 2 > 3 * prediction_var] , y_pred[residual ** 2 > 3 * prediction_var] ] 
#inliers = [  y[residual ** 2 < 3 * prediction_var] , y_pred[residual ** 2 < 3 * prediction_var] ]
#plt.plot(*outliers, linestyle = 'None', marker='o', color = 'red', alpha = 0.4)
#plt.plot(*inliers, linestyle = 'None', marker='o', color = 'blue', alpha = 0.4)
#plt.xlabel('y')
#plt.ylabel('y_pred')
#plt.title('outliers - all ')
#plt.show()

################ Applying to test dataset ####################

X_test = df_test.values
y_pred = model_maker.xgb_best_fit.predict(X_test)

y_pred = np.exp(y_pred)

df_pred = pd.DataFrame(
	y_pred
	,index =df_test.index 
	,columns=['SalePrice'] 
	)

df_pred.to_csv('SalePrice_predictions.csv')

df_salePrice.hist()
df_pred.hist('SalePrice')
