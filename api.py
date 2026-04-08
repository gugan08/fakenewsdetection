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
              "not", "but", "or", "at", "by", "from", "as", "do", "if", "no"}
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

    # Calculate confidence
    confidence = 50.0
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vec_input)[0][1] * 100
    elif hasattr(model, "decision_function"):
        dist = model.decision_function(vec_input)[0]
        confidence = (1 / (1 + math.exp(-dist))) * 100

    # Threshold: > 50% → Real
    if confidence > 50.0:
        return PredictionResponse(prediction=1, label_string="Mostly REAL News", confidence=confidence)
    else:
        return PredictionResponse(prediction=0, label_string="Likely FAKE News", confidence=confidence)
