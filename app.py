
from flask import Flask, jsonify, request, make_response
import nlp as n
from flask_cors import CORS , cross_origin
from flask import json
import numpy as np
import pandas as pd
import re
import nltk
import pickle
import joblib
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import json

app= Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/report/analysis": {"origins": "*"}})


cv = CountVectorizer(max_features = 1420)
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')



 

@app.route("/report/analysis",methods=["POST"])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def main():

    json_data = request.json

    a_value = json_data["detailedreport"]

    corpus=[]


    review = re.sub('[^a-zA-Z]', ' ', a_value)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

    cv = pickle.load(open('./bow.pkl',"rb"))
    xfresh = cv.transform(corpus).toarray()

    rfc = joblib.load("./model")
    result = rfc.predict(xfresh)[0]

    response = make_response(
    jsonify(
        {"prediction":result}
    ),
    200,
    )
    response.headers["Content-Type"] = "application/json"
    return response
    

if __name__=="__main__":
    app.run(debug=True)