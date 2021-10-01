from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# df=pd.read_csv("cluster_house123.csv")
# model=df.drop(['ID'],axis=1)

app=Flask(__name__)
model=pickle.load(open('regressor.pkl','rb'))
model1=pickle.load(open('kmeans1.pkl','rb'))

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/form')
def form():
    return render_template('home1.html')


@app.route('/form1')
def form1():
    return render_template('home.html')

@app.route('/form2')
def form2():
    return render_template('chart.html')


@app.route('/predict_type',methods=['POST'])
def predict_type():

    data12=request.form['val12']
    data13=request.form['val13']
    data14=request.form['val14']
    data15=request.form['val15']
    data16=request.form['val16']
    data17=request.form['val17']
    data18=request.form['val18']
    arr=np.array([[data12,data13,data14,data15,data16,data17,data18]])
    pred=model1.predict(arr)

    return render_template('after.html',data=pred,bedrooms=data12,bathrooms=data13,living=data14,lot=data15,above=data16,basement=data17,floors=data18)


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0]) 
    return render_template('home1.html', prediction_text='No of floors in the house is :{}'.format(output),bedrooms=int_features[0],bathrooms=int_features[1],living=int_features[2],lot=int_features[3],above=int_features[4],basement=int_features[5])

if __name__=="__main__":
    app.run(debug=True)