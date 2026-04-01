from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import math

app = FastAPI(title="Fake News Detection API")

# Allow CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print("Warning: Models not found or failed to load. Run train_model.py first.")
    model = None
    vectorizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    label_string: str
    confidence: float

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

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    cleaned_text = preprocess(request.text)
    vectorized_input = vectorizer.transform([cleaned_text])
    
    # Calculate Probability (Confidence mapping)
    if hasattr(model, "predict_proba"):
        real_prob = model.predict_proba(vectorized_input)[0][1]
        confidence = real_prob * 100
    elif hasattr(model, "decision_function"):
        dist = model.decision_function(vectorized_input)[0]
        confidence = (1 / (1 + math.exp(-dist))) * 100
    else:
        confidence = 50.0  # Fallback

    # Apply Explicit Rule: > 50% = Real
    if confidence > 50.0:
        prediction_val = 1
        label_str = "Mostly REAL News"
    else:
        prediction_val = 0
        label_str = "Likely FAKE News"
        
    return PredictionResponse(
        prediction=prediction_val,
        label_string=label_str,
        confidence=confidence
    )

@app.get("/")
def read_root():
    return {"message": "Fake News API is running. Use POST /predict to analyze text."}
