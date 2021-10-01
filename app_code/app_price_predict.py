from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# df=pd.read_csv("cluster_house123.csv")
# model=df.drop(['ID'],axis=1)

app=Flask(__name__)
model=pickle.load(open('gboostmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('price_predict.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():

    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    sqft_living = request.form['sqft_living']
    sqft_lot = request.form['sqft_lot']
    floors = request.form['floors']
    waterfront = request.form['waterfront']
    view = request.form['view']
    condition = request.form['condition']
    grade = request.form['grade']
    sqft_above = request.form['sqft_above']
    sqft_basement = request.form['sqft_basement']
    yr_built = request.form['yr_built']
    yr_renovated = request.form['yr_renovated']
    lat = request.form['lat']
    long = request.form['long']
    yr_sold = request.form['yr_sold']
    arr=np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                            waterfront, view, condition, grade, sqft_above,
                            sqft_basement,yr_built, yr_renovated, lat, long,yr_sold]])
    pred=model.predict(arr)
    output = round(pred[0]) 
    return render_template('price_predict.html', prediction_text='Price of the house is $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
