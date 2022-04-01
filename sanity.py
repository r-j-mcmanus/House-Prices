import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', index_col = 'Id')
df_pred = pd.read_csv('SalePrice_predictions.csv', index_col = 'Id')

df['SalePrice'].hist()
df_pred['SalePrice'].hist()
plt.show()