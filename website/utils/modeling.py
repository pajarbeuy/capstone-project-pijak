import numpy as np
import joblib
import streamlit as st

from utils.paths import MODEL_PATH, VECTORIZER_PATH
from utils.preprocessing import preprocess_text
from utils.ui import LABEL_ID


@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def margin_to_confidence(margins: np.ndarray) -> float:
    scores = np.asarray(margins, dtype=float).reshape(-1)
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum()
    return float(probs.max())


def predict_sentiment(text: str) -> dict:
    model, vectorizer = load_model_assets()
    processed = preprocess_text(text)
    if not processed:
        raise ValueError("Teks terlalu pendek setelah preprocessing.")

    features = vectorizer.transform([processed])
    pred_id = int(model.predict(features)[0])
    margins = model.decision_function(features)
    confidence = margin_to_confidence(margins)

    return {
        "label_id": pred_id,
        "label": LABEL_ID.get(pred_id, str(pred_id)),
        "processed_text": processed,
        "confidence": confidence,
    }


def predict_many(texts: list[str]) -> list[dict]:
    model, vectorizer = load_model_assets()
    processed = [preprocess_text(text) for text in texts]
    safe_processed = [text if text else " " for text in processed]
    features = vectorizer.transform(safe_processed)
    pred_ids = model.predict(features)
    margins = model.decision_function(features)

    results = []
    for pred_id, source, margin in zip(pred_ids, processed, margins):
        results.append(
            {
                "sentiment_predicted": LABEL_ID.get(int(pred_id), str(pred_id)),
                "processed_text": source,
                "confidence": margin_to_confidence(margin),
            }
        )
    return results
