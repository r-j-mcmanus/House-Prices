import pandas as pd 

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print('df length', len(df))
print('df_test length', len(df_test))


print('df_all length', len(df_all),'\n')

from imputing import imputing
from categorical_feature_minipulation import categorical_feature_minipulation

df_all = imputing(df_all)
df_all = categorical_feature_minipulation(df_all)

print(df_all.head(),'\n')


print(df_test_filled.isna().sum(),'\n')



print('check that we have all elemebts')

id_values_1 = set(df_test['Id'].values)
id_values_2 = set(df_test_filled['Id'].values)
print('df_test missing Ids',id_values_1 - id_values_2)
print(df_test[['Id']])
print(df_test_filled[['Id','SalePrice']],'\n')

id_values_1 = set(df['Id'].values)
id_values_2 = set(df_filled['Id'].values)
print('df missing Ids',id_values_1 - id_values_2)
print(df[['Id','SalePrice']])
print(df_filled[['Id','SalePrice']])



#print(df.iloc[-1],'\n')
#print(df_test.iloc[0])