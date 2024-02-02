# dependencies flask,scikit-learn,pandas,pickel-mixin

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
data = pd.read_csv('Housing.csv')
pipe = pickle.load(open("TrainedModel.pkl","rb"))
@app.route('/')
def index():
    furnished=data['furnishingstatus'].unique()
    return render_template('index.html', furnished=furnished)

@app.route('/predict', methods =['POST'])
def predict():
    area = request.form.get('area')
    bedrooms = request.form.get('bedrooms')
    bathrooms = request.form.get('bathrooms')
    stories = request.form.get('stories')
    mainroad = request.form.get('mainroad')
    guestroom = request.form.get('guestrooms')
    basement = request.form.get('basement')
    hotwaterheating = request.form.get('waterheater')
    airconditioning = request.form.get('Ac')
    parking = request.form.get('parking')
    furnishingstatus = request.form.get('furnished')

    print(area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,furnishingstatus)
    input = pd.DataFrame([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,furnishingstatus]], columns= ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','furnishingstatus'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True , port=5001)