import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', index_col = 'Id')
df['log_SalePrice'] = df['SalePrice'].apply(lambda x : np.log(x))
df.drop('SalePrice', axis = 1)

from sklearn.preprocessing import RobustScaler


