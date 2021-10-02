import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import math


df=pd.read_csv("kc_house_data.csv")

#the id,date, column no need for analysis so drop it
df1=df.drop(['id','date','yr_built','yr_renovated', 'zipcode'],axis=1)

cdf = df1[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_above','sqft_basement','floors']]

x = cdf.iloc[:, :6]
y = cdf.iloc[:, -1]


regressor = LinearRegression()
regressor.fit(x, y)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

clf = regressor.fit(X_train,y_train)
y_pred = clf.predict(X_test)


ID=np.array(np.round(y_test))
Prediction=np.array(np.round(y_pred))


accuracy_test = accuracy_score(ID,Prediction)
print("accuracy =",accuracy_test)
print(" ")

mse=mean_squared_error(y_train,regressor.predict(X_train))
mae=mean_absolute_error(y_train,regressor.predict(X_train))
r2=metrics.r2_score(y_train,regressor.predict(X_train))

print("Train mse = ",mse)
print("Train mae = ",mae)
print("Train rmse = ",math.sqrt(mse))
print("Train R-squared = ",r2)
print(" ")

test_mse=mean_squared_error(y_test,regressor.predict(X_test))
test_mae=mean_absolute_error(y_test,regressor.predict(X_test))
test_r2=metrics.r2_score(y_test,regressor.predict(X_test))

print("Test mse = ",test_mse)
print("Test mae = ",test_mae)
print("Test rmse = ",math.sqrt(test_r2))
print("Test R-squared = ",test_r2)

import pickle
pickle.dump(regressor,open('regressor.pkl','wb'))
