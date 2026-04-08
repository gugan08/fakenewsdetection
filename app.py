import streamlit as st
import joblib
import os
import re
import math

# --- NLTK Setup (with graceful fallback) ---
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

# --- Page Config ---
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

# --- Helpers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_models():
    """Load the trained model and vectorizer from disk."""
    try:
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None


def preprocess(text: str) -> str:
    """Clean and tokenize input text, removing stop words."""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def get_confidence(model, vectorized_input) -> float | None:
    """Extract prediction confidence from the model."""
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(vectorized_input)[0][1] * 100
        if hasattr(model, "decision_function"):
            dist = model.decision_function(vectorized_input)[0]
            return (1 / (1 + math.exp(-dist))) * 100
    except Exception:
        pass
    return None


# --- UI ---
st.title("📡 Fake News Detection System")
st.markdown("Enter a news headline or article below to verify its authenticity.")

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ **Model not found!** Run `python train_model.py` first.")
    st.info("Install dependencies: `pip install -r requirements.txt`")
else:
    user_input = st.text_area(
        "📰 News Content", height=200,
        placeholder="Paste the news article or headline here..."
    )

    if st.button("Check Authenticity"):
        if not user_input.strip():
            st.warning("Please enter some text to check.")
        else:
            with st.spinner("Analyzing text patterns using Machine Learning..."):
                cleaned = preprocess(user_input)
                vec_input = vectorizer.transform([cleaned])
                confidence = get_confidence(model, vec_input)

                # Threshold: > 50% confidence → Real
                if confidence is not None:
                    is_real = confidence > 50.0
                else:
                    is_real = model.predict(vec_input)[0] == 1

                if is_real:
                    pct = f" (Confidence: {confidence:.2f}%)" if confidence else ""
                    st.markdown(
                        f'<div class="prediction-card-real">✅ Mostly REAL News{pct}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    pct = f" (Confidence: {100.0 - confidence:.2f}%)" if confidence else ""
                    st.markdown(
                        f'<div class="prediction-card-fake">🚫 Likely FAKE News{pct}</div>',
                        unsafe_allow_html=True,
                    )

st.markdown("---")
st.caption("Note: This system uses a machine learning classifier.")
