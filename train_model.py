import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- NLTK Setup (graceful fallback) ---
STOP_WORDS = {"a", "an", "the", "and", "is", "in", "it", "to", "of", "for",
              "on", "that", "this", "with", "was", "are", "be", "has", "had",
              "not", "but", "or", "at", "by", "from", "as", "do", "if", "no"}
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_text(text: str) -> str:
    """Clean and tokenize input text, removing stop words."""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def generate_mock_data() -> str:
    """Generate a small mock dataset for training."""
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
    data = data * 20  # Duplicate to increase sample size
    csv_path = os.path.join(BASE_DIR, "news_dataset.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def train():
    """Train the Naive Bayes model and save to disk."""
    csv_file = generate_mock_data()

    print("Loading mock dataset...")
    df = pd.read_csv(csv_file)

    print("Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    print("Done! Saved model.pkl and vectorizer.pkl.")


if __name__ == "__main__":
    train()
