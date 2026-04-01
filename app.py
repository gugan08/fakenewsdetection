import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📡", layout="centered")

st.markdown("""
    <style>
    .prediction-card-fake {
        background-color: #ffcccc;
        padding: 20px;
        margin-top: 20px;
        border-radius: 10px;
        color: #990000;
        font-weight: bold;
        text-align: center;
        border-left: 5px solid #990000;
    }
    .prediction-card-real {
        background-color: #ccffcc;
        padding: 20px;
        margin-top: 20px;
        border-radius: 10px;
        color: #006600;
        font-weight: bold;
        text-align: center;
        border-left: 5px solid #006600;
    }
    </style>
""", unsafe_allow_html=True)

try:
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {"a", "the", "and", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with"}
        
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("📡 Fake News Detection System")
st.markdown("Enter a news headline or article below to verify its authenticity.")

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ **Model not found!** Please run `python train_model.py` to train and generate the prediction model.")
    st.info("If you haven't installed dependencies yet, open a terminal in this folder and run `pip install streamlit pandas scikit-learn nltk joblib`")
else:
    user_input = st.text_area("📰 News Content", height=200, placeholder="Paste the news article or headline here...")
    
    if st.button("Check Authenticity"):
        if not user_input.strip():
            st.warning("Please enter some text to check.")
        else:
            with st.spinner("Analyzing text patterns using Machine Learning..."):
                cleaned_text = preprocess(user_input)
                vectorized_input = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_input)[0]
                
                try:
                    if hasattr(model, "predict_proba"):
                        # Get probability of class 1 (Real News)
                        real_prob = model.predict_proba(vectorized_input)[0][1]
                        confidence = real_prob * 100
                    elif hasattr(model, "decision_function"):
                        import math
                        # Raw distance from hyperplane (positive = Real, negative = Fake)
                        dist = model.decision_function(vectorized_input)[0]
                        confidence = (1 / (1 + math.exp(-dist))) * 100
                    else:
                        confidence = None
                except:
                    confidence = None

                # Apply user requested explicit threshold:
                if confidence is not None:
                    if confidence > 50.0:
                        prediction = 1
                    else:
                        prediction = 0

                if prediction == 1:
                    conf_text = f" (Confidence: {confidence:.2f}%)" if confidence else ""
                    st.markdown(f'<div class="prediction-card-real">✅ Mostly REAL News{conf_text}</div>', unsafe_allow_html=True)
                else:
                    conf_text = f" (Confidence: {100.0 - confidence:.2f}%)" if confidence else ""
                    st.markdown(f'<div class="prediction-card-fake">🚫 Likely FAKE News{conf_text}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: This system uses a machine learning classifier.")
