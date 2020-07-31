import os
import psycopg2

from flask import Flask, request
from flask import render_template, jsonify, url_for

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pickle
import datetime
import json

import mnist

app = Flask(__name__)

@app.route('/')
def Homepage():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def Prediction():
    #fetch input from form + loading model
    from_form = request.form['text_field']
    with open('data/news_train.pkl', 'rb') as f:
        news_train = pickle.load(f)
    with open('models/model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('prediction_map.json', 'r') as pred_map:
        prediction_map = json.load(pred_map)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    cv_fit = count_vect.fit_transform(news_train.data)
    X_train_tfidf = tfidf_transformer.fit_transform(cv_fit)

    count_vect_data = count_vect.transform([from_form])
    tfidf_transformer_data = tfidf_transformer.transform(count_vect_data)
    prediction = clf.predict(tfidf_transformer_data)
    prediction_name = prediction_map.get(str(prediction[0]), "couldn't find name")
  
    response = {
        'status': 200,
        'prediction':prediction_name,
        'created_at': datetime.datetime.now()
    }
    return jsonify(response)

@app.route('/mnist')
def Mnist():
    # mnist.mnist()
    # conn = psycopg2.connect(host="ec2-54-75-244-161.eu-west-1.compute.amazonaws.com",database="d790i0bj2ikkeq",user="qumbrzpxinjjas",password="157959a5cf68334fcb38a2e3df6f65b97e82186eaa9c21ce6e80c6e1a4aff253")
    # cur = conn.cursor()
    
    # cur.execute('select model from models')
    # bytes_model = cur.fetchone()
    # model = pickle.loads(bytes_model[0])
    # print(model.get_params())
        
    # close communication with the PostgreSQL database server
    # cur.close()
    # commit the changes
    # conn.commit()

    return render_template('mnist.html')

@app.route('/fashion_mnist')
def Fashion_mnist():
    return render_template('fashion_mnist.html')

if __name__=='__main__':
   app.run(debug=True)
