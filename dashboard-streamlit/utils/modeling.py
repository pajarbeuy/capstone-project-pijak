import numpy as np
import joblib
import streamlit as st
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences

from utils.paths import MODEL_PATH, VECTORIZER_PATH
from utils.preprocessing import preprocess_text
from utils.ui import LABEL_TEXT


@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = load_model(str(MODEL_PATH))
    tokenizer = joblib.load(str(VECTORIZER_PATH))
    return model, tokenizer


def predict_sentiment(text: str) -> dict:
    model, tokenizer = load_model_assets()
    processed = preprocess_text(text)
    if not processed:
        raise ValueError("Teks terlalu pendek setelah preprocessing.")

    # Convert text to sequences and pad
    sequences = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequences, maxlen=100)
    
    # Get predictions (probabilities)
    probs = model.predict(padded, verbose=0)[0]
    pred_label = str(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "label": pred_label,
        "label_text": LABEL_TEXT.get(pred_label, pred_label.title()),
        "processed_text": processed,
        "confidence": confidence,
    }


def predict_many(texts: list[str]) -> list[dict]:
    model, tokenizer = load_model_assets()
    processed = [preprocess_text(text) for text in texts]
    safe_processed = [text if text else " " for text in processed]
    
    # Convert texts to sequences and pad
    sequences = tokenizer.texts_to_sequences(safe_processed)
    padded = pad_sequences(sequences, maxlen=100)
    
    # Get predictions
    probs = model.predict(padded, verbose=0)
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

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
