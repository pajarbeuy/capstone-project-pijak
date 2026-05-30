from collections import Counter

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
    ).generate(text)


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

    x_train_tfidf = vectorizer.transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    y_train_pred = model.predict(x_train_tfidf)
    y_pred = model.predict(x_test_tfidf)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "metric"})

    labels = ["negative", "neutral", "positive"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    metadata = {
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_accuracy": float((y_train_pred == y_train).mean()),
        "best_params": {"svm__C": 1, "svm__kernel": "linear"},
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "tfidf": {
            "max_features": 10000,
            "ngram_range": "(1, 2)",
            "min_df": 17,
            "max_df": 0.8,
            "sublinear_tf": False,
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
