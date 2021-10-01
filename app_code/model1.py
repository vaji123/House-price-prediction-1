import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

df=pd.read_csv("kc_house_data.csv")

#the id,date, column no need for analysis so drop it
df1=df.drop(['id','date','yr_built','yr_renovated', 'zipcode'],axis=1)

cdf = df1[['bedrooms','bathrooms','floors','sqft_living','sqft_lot','sqft_above','sqft_basement','condition']]

x = cdf.iloc[:, :7]
y = cdf.iloc[:, -1]

kmean=KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmean.fit(x, y)



import pickle
pickle.dump(kmean,open('kmeans.pkl','wb'))

