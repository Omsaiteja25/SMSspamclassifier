import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import sklearn


stemmer = PorterStemmer()

def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    a = []
    for i in text:
        if i.isalnum():
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        a.append(stemmer.stem(i))

    return ' '.join(a)

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
model = pickle.load(open('final_model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('predict'):
    # we need to perform 4 steps here.
    # 1. Preprocess text
    transformed_sms = text_preprocessing(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)
    # 4. Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')