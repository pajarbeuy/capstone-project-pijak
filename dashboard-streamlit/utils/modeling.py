import numpy as np
import joblib
import streamlit as st

from utils.paths import MODEL_PATH, VECTORIZER_PATH
from utils.preprocessing import preprocess_text
from utils.ui import LABEL_TEXT


@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = joblib.load(str(MODEL_PATH))
    vectorizer = joblib.load(str(VECTORIZER_PATH))
    return model, vectorizer


def predict_sentiment(text: str) -> dict:
    model, vectorizer = load_model_assets()
    processed = preprocess_text(text)
    if not processed:
        raise ValueError("Teks terlalu pendek setelah preprocessing.")

    # Vectorize text using TF-IDF
    try:
        text_vector = vectorizer.transform([processed])
    except Exception as e:
        raise ValueError(f"Error saat vectorization: {str(e)}. Pastikan preprocessing konsisten.")
    
    # Get predictions
    try:
        pred_label = str(model.predict(text_vector)[0])
    except Exception as e:
        raise ValueError(f"Error saat prediksi: {str(e)}")
    
    # Calculate confidence
    try:
        pred_proba = model.predict_proba(text_vector)[0]
        confidence = float(np.max(pred_proba))
    except AttributeError:
        # For SVC without probability=True, use decision_function
        try:
            decision = model.decision_function(text_vector)
            # decision shape is (1, n_classes) for multi-class or (1,) for binary
            if decision.ndim > 1:
                decision = np.max(np.abs(decision[0]))
            else:
                decision = np.abs(decision[0])
            confidence = float(1 / (1 + np.exp(-decision)))  # Sigmoid normalization
        except Exception:
            confidence = 0.5

    return {
        "label": pred_label,
        "label_text": LABEL_TEXT.get(pred_label, pred_label.title()),
        "processed_text": processed,
        "confidence": confidence,
    }


def predict_many(texts: list[str]) -> list[dict]:
    model, vectorizer = load_model_assets()
    processed = [preprocess_text(text) for text in texts]
    safe_processed = [text if text else " " for text in processed]
    
    # Vectorize texts using TF-IDF
    try:
        text_vectors = vectorizer.transform(safe_processed)
    except Exception as e:
        raise ValueError(f"Error saat vectorization: {str(e)}")
    
    # Get predictions
    try:
        pred_labels = model.predict(text_vectors)
    except Exception as e:
        raise ValueError(f"Error saat prediksi batch: {str(e)}")
    
    # Calculate confidences
    try:
        pred_probas = model.predict_proba(text_vectors)
        confidences = np.max(pred_probas, axis=1)
    except AttributeError:
        # For SVC without probability=True, use decision_function
        try:
            decisions = model.decision_function(text_vectors)
            # decisions shape is (n_samples, n_classes) for multi-class or (n_samples,) for binary
            if decisions.ndim > 1:
                # Multi-class: take max absolute value per sample
                confidences = np.max(np.abs(decisions), axis=1)
            else:
                # Binary: take absolute value
                confidences = np.abs(decisions)
            confidences = 1 / (1 + np.exp(-confidences))  # Sigmoid normalization
        except Exception:
            confidences = np.full(len(pred_labels), 0.5)

    results = []
    for pred_label, source, conf in zip(pred_labels, processed, confidences):
        results.append(
            {
                "sentiment_predicted": str(pred_label),
                "processed_text": source,
                "confidence": float(conf),
            }
        )
    return results
