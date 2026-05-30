import json

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.paths import DATA_PATH, METADATA_PATH


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
# Kolom teks akhir hasil preprocessing notebook (setelah stopword removal)
TEXT_FEATURE_COL = "text_akhir"
# Kolom label sentimen di CSV
TARGET_COL = "polarity"


def prepare_notebook02_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standarisasi nama kolom agar konsisten di seluruh dashboard.

    CSV asli: text_akhir (teks sudah diproses), polarity (label sentimen)
    Dashboard: review_text_stemmed (alias), sentiment_label (alias)
    """
    df = df.copy()

    # Gunakan 'polarity' sebagai sentiment_label utama
    df["sentiment_label"] = df[TARGET_COL].astype(str).str.strip()
    df["sentiment_encoded"] = df["sentiment_label"].map(LABEL_TO_ID).astype("Int64")

    # Alias kolom teks akhir agar konsisten dengan kode dashboard lama
    df["review_text_stemmed"] = df[TEXT_FEATURE_COL].fillna("").astype(str)

    if "review_word_count" not in df.columns:
        df["review_word_count"] = df["review_text_stemmed"].str.split().str.len()
    return df


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return prepare_notebook02_dataset(pd.read_csv(DATA_PATH))


@st.cache_data(show_spinner=False)
def load_modeling_dataset() -> pd.DataFrame:
    return load_dataset().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_modeling_dataset()
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment_label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["sentiment_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    order = {"positive": 0, "neutral": 1, "negative": 2}
    return counts.sort_values("sentiment", key=lambda s: s.map(order)).reset_index(drop=True)
