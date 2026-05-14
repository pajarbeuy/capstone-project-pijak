import json

import pandas as pd
import streamlit as st

from utils.paths import DATA_PATH, METADATA_PATH, TEST_PATH, TRAIN_PATH


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_modeling_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=["review_text_stemmed"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


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
