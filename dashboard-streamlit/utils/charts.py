from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

from utils.modeling import load_model_assets
from utils.ui import LABEL_COLORS


PLOTLY_TEMPLATE = "plotly_dark"


def sentiment_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=LABEL_COLORS,
        template=PLOTLY_TEMPLATE,
        text_auto=True,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Review")
    return fig


def sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        hole=0.58,
        color="sentiment",
        color_discrete_map=LABEL_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False)
    return fig


def text_length_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="review_word_count",
        nbins=45,
        color="sentiment_label",
        color_discrete_map=LABEL_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Jumlah Kata", yaxis_title="Jumlah Review")
    return fig


def category_chart(df: pd.DataFrame) -> go.Figure:
    top = df["product_category"].value_counts().head(10).reset_index()
    top.columns = ["category", "count"]
    fig = px.bar(
        top.sort_values("count"),
        x="count",
        y="category",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color="count",
        color_continuous_scale=["#1E293B", "#00ADB5"],
    )
    fig.update_layout(xaxis_title="Jumlah Review", yaxis_title="", coloraxis_showscale=False)
    return fig


@st.cache_data(show_spinner=False)
def word_frequencies(df: pd.DataFrame, label: str | None = None, top_n: int = 25) -> pd.DataFrame:
    source = df if label is None else df[df["sentiment_label"] == label]
    words = " ".join(source["review_text_stemmed"].dropna().astype(str)).split()
    counts = Counter(words).most_common(top_n)
    return pd.DataFrame(counts, columns=["word", "count"])


def frequency_chart(freq_df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(
        freq_df.sort_values("count"),
        x="count",
        y="word",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color="count",
        color_continuous_scale=["#1E293B", "#00ADB5"],
        title=title,
    )
    fig.update_layout(xaxis_title="Frekuensi", yaxis_title="", coloraxis_showscale=False)
    return fig


@st.cache_data(show_spinner=False)
def build_wordcloud(df: pd.DataFrame, label: str | None = None):
    if WordCloud is None:
        raise RuntimeError("Library wordcloud belum terpasang. Jalankan `pip install wordcloud`.")

    source = df if label is None else df[df["sentiment_label"] == label]
    text = " ".join(source["review_text_stemmed"].dropna().astype(str))
    if not text.strip():
        text = "sentimen review produk"
    return WordCloud(
        width=1200,
        height=520,
        background_color="#0F172A",
        colormap="viridis",
        max_words=120,
        collocations=False,
    ).generate(text).to_array()


@st.cache_data(show_spinner=False)
def model_performance(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model, vectorizer = load_model_assets()
    x = df_clean["review_text_stemmed"].fillna("").astype(str).values
    y = df_clean["sentiment_label"].astype(str).values

    x_train, x_test, y_train, y_true = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Vectorize using TF-IDF
    try:
        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)
    except Exception as e:
        raise ValueError(f"Error saat vectorization: {str(e)}")
    
    # Get predictions
    try:
        y_train_pred_encoded = model.predict(x_train_vec)
        y_pred_encoded = model.predict(x_test_vec)
    except Exception as e:
        raise ValueError(f"Error saat prediksi: {str(e)}")
    
    # Convert predictions to string labels
    y_train_pred = np.array([str(label).strip() for label in y_train_pred_encoded])
    y_train_clean = np.array([str(label).strip() for label in y_train])
    y_pred = np.array([str(label).strip() for label in y_pred_encoded])
    y_true_clean = np.array([str(label).strip() for label in y_true])

    report = classification_report(y_true_clean, y_pred, output_dict=True, zero_division=0)
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            report_data.append({
                "metric": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"]
            })
    report_df = pd.DataFrame(report_data)

    labels = ["negative", "neutral", "positive"]
    matrix = confusion_matrix(y_true_clean, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    # Hitung accuracy dari weighted avg f1-score (sama dengan accuracy untuk multiclass)
    test_accuracy = float(report.get("accuracy", report["weighted avg"]["f1-score"]))
    metadata = {
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "test_accuracy": test_accuracy,
        "train_accuracy": float((y_train_pred == y_train_clean).mean()),
        "best_params": {"kernel": "linear", "C": 1.0},
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "svm_config": {
            "vectorizer": "TfidfVectorizer",
            "max_features": "default",
            "kernel": "linear",
        },
    }
    return report_df, matrix_df, metadata


def confusion_matrix_chart(matrix_df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale=["#1E293B", "#00ADB5"],
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Prediksi", yaxis_title="Aktual", coloraxis_showscale=False)
    return fig
