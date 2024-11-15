from flask import Flask,request, render_template
import numpy as np
import os
import pickle
import joblib

#loading models
# dtr = pickle.load(open('dtr.pkl','rb'))
# preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#save the model
# joblib.dump(dtr,'dtr.pkl')
# joblib.dump(preprocessor,'preprocessor.pkl')

#load the model
dtr = joblib.load('templates/dtr.pkl')
preprocessor = joblib.load('templates/preprocessor.pkl')

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('indexfolder.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('indexfolder.html',prediction = prediction)

if __name__=="__main__":
    print("Starting the flask server")
    app.run(debug=True, port=5001)