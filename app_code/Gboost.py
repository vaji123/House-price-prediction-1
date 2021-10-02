# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# evaluation on test data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv("cleaned_kc.csv")

print(df.columns.values)

# Features
X = df.drop('price',axis=1)

# Label
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# everything has been scaled between 1 and 0
print('Max: ',X_train.max())
print('Min: ', X_train.min())

#FINDING ACCURACY FOR MULTIPLE LINEAR REGRESSION MODEL

#from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

#lr = LinearRegression()
#lr.fit(X_train,y_train)
#lr_score = lr.score(X_train, y_train)
#pred_lr = lr.predict(X_test)
#expl_lr = explained_variance_score(pred_lr,y_test)



#calculate model score
##print("Multiple Linear Regression Model Score is ",round(lr.score(X_test,y_test)*100))


#Use K Fold cross validation to measure accuracy of our LinearRegression model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

#cross_val_score(LinearRegression(), X, y, cv=cv)


#Model 2 - XG Boost REGRESSION MODEL
from sklearn.ensemble import GradientBoostingRegressor
gboost = GradientBoostingRegressor()

gboost.fit(X_train, y_train)

pred_gboost = gboost.predict(X_test)

expl_gboost = explained_variance_score(pred_gboost,y_test)


#calculate model score
#print("Multiple Linear Regression Model Score is ",round(gboost.score(X_test,y_test)*100))


cross_val_score(GradientBoostingRegressor(), X, y, cv=cv)


# predictions on the test set
predictions = gboost.predict(X_test)

print('MAE: ',mean_absolute_error(y_test,predictions))
print('MSE: ',mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)))
print('Variance Regression Score: ',explained_variance_score(y_test,predictions))

print('\n\nDescriptive Statistics:\n',df['price'].describe())
print(df.columns)

import pickle
pickle.dump(gboost,open('gboostmodel.pkl','wb'))