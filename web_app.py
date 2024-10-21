# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sklearn
import numpy as np
import pickle
import joblib
import streamlit as st

# loading the saved model
model = pickle.load(open("C:/Users/hatsh/Downloads/trained_model.sav", 'rb'))
vectorizer = pickle.load(open("C:/Users/hatsh/Downloads/vectorizer.pkl", 'rb'))

# title and instructions
st.title('Fake News Detection')
st.write("Enter the news article text below to classify whether it is 'Real' or 'Fake'.")

# Text input from the user
news_text = st.text_area("News Article Text", height=300)

# Prediction logic when user clicks "Classify"
if st.button("Classify"):
    if news_text:  # If there's text input
        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([news_text])
        
        # Make a prediction using the loaded model
        prediction = model.predict(transformed_text)
        result = 'Fake' if prediction[0] == 0 else 'Real'
        
        # Display the result
        st.success(f'The news is **{result}**.')
    else:
        st.error("Please enter some text to classify.")