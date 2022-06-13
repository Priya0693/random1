# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:36:59 2022

@author: priya
"""
from flask import Flask, render_template, request, jsonify
import nltk
#import nltk
#nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('D:/Download files/model deploy/random/model2 (2).pkl', 'rb'))
tfidfvect = pickle.load(open('D:/Download files/model deploy/random/tfidfvect2 (3).pkl', 'rb'))

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'The news 📰 is FAKE ' if model.predict(review_vect) == 0 else 'The news 📰 is REAL'
    return prediction

def main():
    st.title('fake news prediction')
    
    text = st.text_input('Input your text here')
    
    
    #code for prediction
    prediction = ''
    
    if st.button('news result'):
        prediction =predict([text])
        
    st.success(prediction)
    
    
if __name__ == '__main__':
    main()
    