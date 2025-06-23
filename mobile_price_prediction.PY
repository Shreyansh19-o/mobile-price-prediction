import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv(r'train_data.csv')

x=df.iloc[:,0:-1].values #here [:,0:-1] first : represents all rows and after from column index 0 to -1(last column)
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_training,x_testing,y_training,y_testing=train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression 
reg=LinearRegression()
reg.fit(x_training,y_training)

pred_y=reg.predict(x_testing)

#feature selection backward elimination
import statsmodels.api as sm
x=np.append(arr=np.ones((1800,1)).astype(int),values=x,axis=1) 
#adding a new column (1800,1) no. of rows 
x_opt=x[:,[0,3,4,5,6,15,16,17,18,19,20,21]]
#variable x_opt = [:,[1...21]] all rows and all columns except the last predection column
regressor_OLS=sm.OLS(endog=y,exog=x).fit() #endog put value we want as output and in endog value as input
print(regressor_OLS.summary()) 
#now check the summary then check all p values which are then sl=0.05 those values which are higher then sl remove them from the list 
