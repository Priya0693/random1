from flask import Flask, render_template, request, jsonify
import nltk
#import nltk
#nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

import pickle

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('model2 (2).pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2 (3).pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'The prediction for this news 📰 is FAKE ' if model.predict(review_vect) == 0 else 'the prediction for this news 📰 is REAL'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
