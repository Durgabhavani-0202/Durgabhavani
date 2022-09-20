# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:18:34 2022

@author: user
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
file_name = 'res_review.pkl'
classifier = pickle.load(open(file_name, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    str = request.form['Review']
    data = [str]
    vect = cv.transform(data).toarray()
    myprediction = classifier.predict(vect)


    return render_template('result.html', prediction= myprediction)

if __name__ == "__main__":
    app.run(debug=True)