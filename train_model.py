import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {"a", "the", "and", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with"}
        
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def generate_mock_data():
    data = [
        {"text": "Scientists discover new planet capable of sustaining life.", "label": 1},
        {"text": "Local man eats 50 hotdogs in record time, crowd goes wild.", "label": 1},
        {"text": "Government announces tax cuts for middle class families.", "label": 1},
        {"text": "New study shows coffee improves long-term memory.", "label": 1},
        {"text": "Technology giant launches revolutionary new smartphone.", "label": 1},
        
        {"text": "Aliens land on Earth and demand to speak to the manager.", "label": 0},
        {"text": "Breaking: Eating rocks cures all known diseases instantly.", "label": 0},
        {"text": "Secret society of cats secretly controls the global economy.", "label": 0},
        {"text": "Moon is actually made of cheese, NASA whistleblower claims.", "label": 0},
        {"text": "Drinking bleach is the best way to stay healthy, says internet doctor.", "label": 0},
    ]
    data = data * 20 
    df = pd.DataFrame(data)
    df.to_csv('news_dataset.csv', index=False)
    return 'news_dataset.csv'

def train():
    download_nltk_data()
    
    # Always forcefully generate the mock CSV so it overrides the huggingface one if it existed
    csv_file = generate_mock_data()
        
    print("Loading V1 Generalized Mock Data...")
    df = pd.read_csv(csv_file)

    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Generalized Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model, 'model.pkl')
    print("Rollback to V1 Mock Model Complete! Saved model.pkl and vectorizer.pkl.")

if __name__ == "__main__":
    train()
