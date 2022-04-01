import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer

from imputing import imputing

if __name__ == '__main__':
	pd.set_option('display.max_columns', None)

	df = pd.read_csv('train.csv', index_col = 'Id')
	df_test = pd.read_csv('test.csv', index_col = 'Id')
	df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
	df.drop('SalePrice', axis = 1)

	df = imputing(df)

	print(df.info())

	print('We will have a lot of catigorical features, we we should look into them.','\n\n')

	################# GarageType ######################

	df_temp = pd.get_dummies(df[['log_SalePrice','GarageType']])

	print(df_temp.info())

	print([ [col, df_temp[df_temp[col] == 1][col].sum()] for col in df_temp ])
	print('we see that GarageType_2Types, GarageType_CarPort, GarageType_Basment all only have a couple samples','\n')


	print(df_temp[df_temp['GarageType_2Types'] == 1]['log_SalePrice'].describe(),'\n')
	print(df_temp[df_temp['GarageType_CarPort'] == 1]['log_SalePrice'].describe(),'\n')
	print(df_temp[df_temp['GarageType_Basment'] == 1]['log_SalePrice'].describe(),'\n')

	print('All three have simmiler stat properies wrt log_salePrice, so we can combine them','\n\n')

	################# MSZoning ######################

	df = df.replace('C (all)', 'C')
	print(df['MSZoning'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','MSZoning']])
	print('we see that most values are in Rl, while very few are in FV, RH or C\n')

	print('MSZoning_RL\n',df_temp[df_temp['MSZoning_RL'] == 1]['log_SalePrice'].describe(),'\n')
	print('MSZoning_RM\n',df_temp[df_temp['MSZoning_RM'] == 1]['log_SalePrice'].describe(),'\n')
	print('MSZoning_FV\n',df_temp[df_temp['MSZoning_FV'] == 1]['log_SalePrice'].describe(),'\n')
	print('MSZoning_RH\n',df_temp[df_temp['MSZoning_RH'] == 1]['log_SalePrice'].describe(),'\n')
	print('MSZoning_C\n',df_temp[df_temp['MSZoning_C'] == 1]['log_SalePrice'].describe(),'\n')

	print('the mean FV is quiet a lot higer than RH or C with respect to their variances, so we will combine RH and C, but leave FV')

	################# MSZoning ######################

	print(df['Street'].value_counts())


	df_temp = pd.get_dummies(df[['log_SalePrice','Street']])
	print('Street_Pave\n',df_temp[df_temp['Street_Pave'] == 1]['log_SalePrice'].describe(),'\n')
	print('Street_Grvl\n',df_temp[df_temp['Street_Grvl'] == 1]['log_SalePrice'].describe(),'\n')

	print('grvl only has 6 counts so might lead to over fitting') 
	print('but also its distribution of log sale price is lower')
	print('the 25% quartile of pave is higehr than the 50% of grvl')
	print('so well keep it\n\n')

	################# Alley ######################

	print(df['Alley'].value_counts())
	print('Alley seems fine')

	################# LotShape ######################

	print(df['LotShape'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','LotShape']])
	print('LotShape_IR2\n',df_temp[df_temp['LotShape_IR2'] == 1]['log_SalePrice'].describe(),'\n')
	print('LotShape_IR3\n',df_temp[df_temp['LotShape_IR3'] == 1]['log_SalePrice'].describe(),'\n')

	print('These two lot shapes have simmilar properies wrt sale price so we combine them')

	################# LandContour ######################

	print(df['LandContour'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','LandContour']])
	print('LandContour_Bnk\n',df_temp[df_temp['LandContour_Bnk'] == 1]['log_SalePrice'].describe(),'\n')
	print('LandContour_HLS\n',df_temp[df_temp['LandContour_HLS'] == 1]['log_SalePrice'].describe(),'\n')
	print('LandContour_Low\n',df_temp[df_temp['LandContour_Low'] == 1]['log_SalePrice'].describe(),'\n')

	print('Seems fine, maybe combine LandContour_HLS and LandContour_Low')

	################# Utilities ######################

	print(df['Utilities'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','Utilities']])
	print('Utilities_AllPub\n',df_temp[df_temp['Utilities_AllPub'] == 1]['log_SalePrice'].describe(),'\n')
	print('Utilities_NoSeWa\n',df_temp[df_temp['Utilities_NoSeWa'] == 1]['log_SalePrice'].describe(),'\n')

	print('drop Utilities_NoSeWa\n')

	################# LotConfig ######################

	print(df['LotConfig'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','LotConfig']])
	print('LotConfig_FR3\n',df_temp[df_temp['LotConfig_FR3'] == 1]['log_SalePrice'].describe(),'\n')

	print('drop LotConfig_FR3\n')

	################# LandSlope ######################

	print(df['LandSlope'].value_counts())
	print('drop LandSlope_Sev\n')


	################# Neighborhood ######################

	print(df['Neighborhood'].value_counts())
	print('drop Neighborhood_Blueste\n')

	################# Condition1 ######################

	print(df['Condition1'].value_counts())
	df_temp = pd.get_dummies(df[['log_SalePrice','Condition1']])

	print(df_temp.info())
	print('Condition1_RRAe\n',df_temp[df_temp['Condition1_RRAe'] == 1]['log_SalePrice'].describe(),'\n')
	print('Condition1_PosA\n',df_temp[df_temp['Condition1_PosA'] == 1]['log_SalePrice'].describe(),'\n')
	print('Condition1_RRNn\n',df_temp[df_temp['Condition1_RRNn'] == 1]['log_SalePrice'].describe(),'\n')
	print('Condition1_RRNe\n',df_temp[df_temp['Condition1_RRNe'] == 1]['log_SalePrice'].describe(),'\n')


	print('combine Condition1_PosA Condition1_RRNn Condition1_RRNe \n')

	################# Condition2 ######################

	print(df['Condition2'].value_counts())
	df_temp = pd.get_dummies(df[['log_SalePrice','Condition1']])
	print(df_temp.info())

	print('combine all that mess \n')

	################# BldgType ######################

	print(df['BldgType'].value_counts())
	print('fine  \n')

	################# RoofStyle ######################

	print(df['RoofStyle'].value_counts())

	print('fine  \n') 

	################# RoofMatl ######################

	print(df['RoofMatl'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','RoofMatl']])

	print(df_temp.info())
	print('RoofMatl_Metal\n',df_temp[df_temp['RoofMatl_Metal'] == 1]['log_SalePrice'],'\n')
	print('RoofMatl_Membran\n',df_temp[df_temp['RoofMatl_Membran'] == 1]['log_SalePrice'],'\n')
	print('RoofMatl_Roll\n',df_temp[df_temp['RoofMatl_Roll'] == 1]['log_SalePrice'],'\n')
	print('RoofMatl_ClyTile\n',df_temp[df_temp['RoofMatl_ClyTile'] == 1]['log_SalePrice'],'\n')

	print('combine them\n')

	################# Exterior1st ######################

	print(df['Exterior1st'].value_counts())
	print(df['Exterior2nd'].value_counts())

	################# Exterior1st ######################

	print(df['MasVnrType'].value_counts())
	print('fine')

	################# ExterQual ######################

	print(df['ExterQual'].value_counts())
	print('fine')

	################# cond ######################

	print(df['ExterQual'].value_counts())
	print('fine')

	################# Foundation ######################

	print(df['Foundation'].value_counts())

	df_temp = pd.get_dummies(df[['log_SalePrice','Foundation']])

	print(df_temp.info())
	print('Foundation_Stone\n',df_temp[df_temp['Foundation_Stone'] == 1]['log_SalePrice'].describe(),'\n')
	print('Foundation_Wood\n',df_temp[df_temp['Foundation_Wood'] == 1]['log_SalePrice'].describe(),'\n')

	print('fine')

	################# BsmtCond ######################

	print(df['BsmtCond'].value_counts())
	print('drop po')

	################# BsmtCond ######################

	print(df['BsmtExposure'].value_counts())
	print('fine')

	################# BsmtFinType1 ######################

	print(df['BsmtFinType1'].value_counts())
	print('BsmtFinType1')

	################# BsmtFinType2 ######################

	print(df['BsmtFinType2'].value_counts())
	print('BsmtFinType2')

	################# Heating ######################

	print(df['Heating'].value_counts())
	print('fine')

	################# HeatingQC ######################

	print(df['HeatingQC'].value_counts())
	print('drop po')

	################# CentralAir ######################

	print(df['CentralAir'].value_counts())
	print('fine')

	################# Electrical ######################

	print(df['Electrical'].value_counts())
	print('drop mix')

	################# KitchenQual ######################

	print(df['KitchenQual'].value_counts())
	print('fine')

	################# Functional ######################

	print(df['Functional'].value_counts())
	print('drop sev')

	################# FireplaceQu ######################

	print(df['FireplaceQu'].value_counts())
	print('fine')

	################# GarageType ######################

	print(df['GarageType'].value_counts())
	print('fine')

	################# GarageFinish ######################

	print(df['GarageFinish'].value_counts())
	print('fine')

	################# GarageQual ######################

	print(df['GarageQual'].value_counts())
	print('fine')

	################# GarageCond ######################

	print(df['GarageCond'].value_counts())
	print('fine')

	################# PavedDrive ######################

	print(df['PavedDrive'].value_counts())
	print('fine')


	################# Fence ######################

	print(df['Fence'].value_counts())
	print('fine')

	################# MiscFeature ######################

	print(df['MiscFeature'].value_counts())
	print('fine')


	################# MiscFeature ######################

	print(df['SaleType'].value_counts())
	print('fine')

	################# MiscFeature ######################

	print(df['SaleCondition'].value_counts())
	print('fine')


################# Applying minipulations ######################

#print('there has to be a better way to do this...')

def Qual_str_to_int_dict_foo(df, col):
	my_dict =  {
		'None' : 0,
		'Po' : df[ df['SalePrice'].notna() & (df[col] == 'Po') ]['SalePrice'].median(),
		'Fa' : df[ df['SalePrice'].notna() & (df[col] == 'Fa') ]['SalePrice'].median(),
		'TA' : df[ df['SalePrice'].notna() & (df[col] == 'TA') ]['SalePrice'].median(),
		'Gd' : df[ df['SalePrice'].notna() & (df[col] == 'Gd') ]['SalePrice'].median(),
		'Ex' : df[ df['SalePrice'].notna() & (df[col] == 'Ex') ]['SalePrice'].median()
	}

	max_val = max(my_dict.values())

	for key in my_dict.keys():
		my_dict[key] = my_dict[key] / max_val

	return my_dict

def categorical_feature_minipulation(df, LandContour_Combine=False):
	df = df.replace('C (all)', 'C')

	#We change features that are catigorical but should be numeric
	#note that we change them into a linear series, but there is no guarantee that they are linearly related to the sale price
	#maybe replcae with the mean sale price for that category?

	Qual_str_to_int_dict = {
		'None' : 0,
		'Po' : 1,
		'Fa' : 2,
		'TA' : 3,
		'Gd' : 4,
		'Ex' : 5,
	}

	Fence_str_to_int_dict = {
		'None' : 0,
		'MnWw' : 1,
		'GdWo' : 2,
		'MnPrv' : 3,
		'GdPrv' : 4
	}

	BsmtExposure_str_to_int_dict = {
		'None' : 0,
		'No' : 1,
		'Mn' : 2,
		'Av' : 3,
		'Gd' : 4
	}

	GarageFinish_str_to_int_dict = {
		'None' : 0,
		'Unf' : 1,
		'RFn' : 2,
		'Fin' : 3
	}

	#print( Qual_str_to_int_dict_foo(df, 'ExterQual') )

	to_change = ['ExterQual','ExterCond','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtQual']
	for col in to_change:
		df[col] = df[col].apply(lambda x : Qual_str_to_int_dict[x])

	df['Fence'] = df['Fence'].apply(lambda x : Fence_str_to_int_dict[x])
	df['BsmtExposure'] = df['BsmtExposure'].apply(lambda x : BsmtExposure_str_to_int_dict[x])
	df['GarageFinish'] = df['GarageFinish'].apply(lambda x : GarageFinish_str_to_int_dict[x])


	#We change features that are numeric but should be catigorical

	df['MoSold_str'] = df['MoSold'].apply(lambda x : str(x))
	df = df.drop('MoSold', axis = 1)

	df['MSSubClass_str'] = df['MSSubClass'].apply(lambda x : str(x))
	df = df.drop('MSSubClass', axis = 1)

	#We split the categorical data
	#we combine small categories as they do not contain much information

	df = pd.get_dummies(df)

	df['GarageType_Other'] = (df['GarageType_2Types'] == 1) | (df['GarageType_CarPort'] == 1) | (df['GarageType_Basment'] == 1)
	df['GarageType_Other'] = df['GarageType_Other'].astype(int)
	df = df.drop(['GarageType_2Types', 'GarageType_CarPort', 'GarageType_Basment'], axis = 1)

	df['MSZoning_Other'] = ((df['MSZoning_RH'] == 1) | (df['MSZoning_C'] == 1))
	df['MSZoning_Other'] = df['MSZoning_Other'].astype(int)
	df = df.drop(['MSZoning_RH', 'MSZoning_C'], axis = 1)

	df['LotShape_Other'] = (df['LotShape_IR2'] == 1) | (df['LotShape_IR3'] == 1)
	df['LotShape_Other'] = df['LotShape_Other'].astype(int)
	df = df.drop(['LotShape_IR2', 'LotShape_IR3'], axis = 1)

	if LandContour_Combine == True:
		df['LandContour_Other'] = (df['LandContour_HLS'] == 1) | (df['LandContour_Low'] == 1)
		df['LandContour_Other'] = df['LandContour_Other'].astype(int)
		df = df.drop(['LandContour_HLS', 'LandContour_Low'], axis = 1)

	df = df.drop('Utilities_NoSeWa', axis = 1)
	df = df.drop('LotConfig_FR3', axis = 1)
	df = df.drop('LandSlope_Sev', axis = 1)
	df = df.drop('Neighborhood_Blueste', axis = 1)

	df['Condition1_Other'] = (df['Condition1_PosA'] == 1) | (df['Condition1_RRNn'] == 1) | (df['Condition1_RRNe'] == 1)
	df['Condition1_Other'] = df['Condition1_Other'].astype(int)
	df = df.drop(['Condition1_PosA', 'Condition1_RRNn', 'Condition1_RRNe'], axis = 1)

	df['Condition2_Other'] = (df['Condition2_Artery'] == 1) | (df['Condition2_Feedr'] == 1) | (df['Condition2_PosA'] == 1) | (df['Condition2_PosN'] == 1) | (df['Condition2_RRAe'] == 1) | (df['Condition2_RRAn'] == 1)  | (df['Condition2_RRNn'] == 1)
	df['Condition2_Other'] = df['Condition2_Other'].astype(int)
	df = df.drop(['Condition2_Artery', 'Condition2_Feedr', 'Condition2_PosA','Condition2_PosN','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn'], axis = 1)

	df['RoofMatl_Other'] = (df['RoofMatl_Metal'] == 1) | (df['RoofMatl_Membran'] == 1) | (df['RoofMatl_Roll'] == 1) | (df['RoofMatl_ClyTile'] == 1)
	df['RoofMatl_Other'] = df['RoofMatl_Other'].astype(int)
	df = df.drop(['RoofMatl_Metal', 'RoofMatl_Membran', 'RoofMatl_Roll','RoofMatl_ClyTile'], axis = 1)

	df = df.drop('Electrical_Mix', axis = 1)
	df = df.drop('Functional_Sev', axis = 1)

	df['MiscFeature_Othr'] = (df['MiscFeature_Othr'] == 1) | (df['MiscFeature_TenC'] == 1) 
	df = df.drop('MiscFeature_TenC', axis = 1)

	df = df.drop('CentralAir_N', axis = 1)
	df = df.drop('PavedDrive_N', axis = 1)
	df = df.drop('Alley_None', axis = 1)

	#print('len(df.columns)',len(df.columns))
	
	return df