import os
import re
import math
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- NLTK Setup (graceful fallback) ---
STOP_WORDS = {"a", "an", "the", "and", "is", "in", "it", "to", "of", "for",
              "on", "that", "this", "with", "was", "are", "be", "has", "had",
              "not", "but", "or", "at", "by", "from", "as", "do", "if", "no",
              "he", "she", "they", "we", "you", "i", "me", "my", "your",
              "his", "her", "its", "our", "their", "what", "which", "who",
              "when", "where", "how", "all", "each", "every", "both", "few",
              "more", "most", "other", "some", "such", "than", "too", "very",
              "can", "will", "just", "should", "now", "also", "into", "only",
              "about", "up", "out", "so", "him", "them", "then", "these",
              "those", "been", "have", "would", "could", "did", "does",
              "said", "were", "over", "after", "before", "between", "own",
              "same", "because", "while", "during", "through"}
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    pass

# --- App ---
app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
except Exception as e:
    print(f"Warning: Models not loaded ({e}). Run train_model.py first.")
    model = None
    vectorizer = None


# --- Schemas ---
class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    label_string: str
    confidence: float


# --- Helpers ---
def preprocess(text: str) -> str:
    """Clean and tokenize input text, removing stop words."""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Fake News API is running. Use POST /predict to analyze text."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned = preprocess(request.text)
    vec_input = vectorizer.transform([cleaned])

    # Get prediction
    prediction = model.predict(vec_input)[0]

    # Calculate confidence using decision function or predict_proba
    confidence = 50.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec_input)[0]
        confidence = proba[1] * 100  # probability of being REAL
    elif hasattr(model, "decision_function"):
        dist = model.decision_function(vec_input)[0]
        # Sigmoid to convert distance to probability
        confidence = (1 / (1 + math.exp(-dist))) * 100

    # Generate label based on prediction and confidence
    if prediction == 1:
        if confidence >= 80:
            label = "Verified REAL News"
        elif confidence >= 65:
            label = "Mostly REAL News"
        else:
            label = "Likely REAL News"
        return PredictionResponse(prediction=1, label_string=label, confidence=confidence)
    else:
        fake_confidence = 100 - confidence
        if fake_confidence >= 80:
            label = "Confirmed FAKE News"
        elif fake_confidence >= 65:
            label = "Likely FAKE News"
        else:
            label = "Possibly FAKE News"
        return PredictionResponse(prediction=0, label_string=label, confidence=confidence)

