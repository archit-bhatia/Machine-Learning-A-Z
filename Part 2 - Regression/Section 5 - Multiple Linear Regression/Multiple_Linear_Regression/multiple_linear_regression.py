#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x=x[:,1:]

#Spliting the data into training and test set
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test results
y_pred=regressor.predict(x_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int) , values =x , axis=1 )
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#Removing x2(index 2)
x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#Removing x1(index 1)
x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#Removing x2(index 4)
x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#Removing x2(index 5) p=0.06
x_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()