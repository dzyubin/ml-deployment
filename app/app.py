from flask import Flask, request
from flask import render_template, jsonify, url_for

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pickle
import datetime
import json

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

if __name__=='__main__':
   app.run(debug=True)