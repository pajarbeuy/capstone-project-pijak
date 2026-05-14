import streamlit as st
import pandas as pd

from utils.charts import confusion_matrix_chart, model_performance
from utils.data import load_modeling_dataset
from utils.ui import page_hero, setup_page


setup_page("Model Performance", "M")
page_hero(
    "Evaluasi Model SVM",
    "Metrik dihitung ulang mengikuti notebook modeling: drop null, split stratified 80:20, TF-IDF, lalu SVM terbaik.",
)

df_clean = load_modeling_dataset()

with st.spinner("Menghitung metrik evaluasi model..."):
    report_df, matrix_df, meta = model_performance(df_clean)

accuracy = float(report_df.loc[report_df["metric"] == "accuracy", "precision"].iloc[0])
macro = report_df.loc[report_df["metric"] == "macro avg"].iloc[0]
weighted = report_df.loc[report_df["metric"] == "weighted avg"].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy Testing", f"{accuracy:.2%}")
col1.caption(f"Training: {meta['train_accuracy']:.2%}")
col2.metric("Precision Macro", f"{macro['precision']:.2%}")
col3.metric("Recall Macro", f"{macro['recall']:.2%}")
col4.metric("F1 Macro", f"{macro['f1-score']:.2%}")

st.caption(
    f"Split modeling: {meta['train_size']:,} data training dan {meta['test_size']:,} data testing. "
    "Dataset memakai `df_clean = df_data.dropna(subset=['review_text_stemmed'])`."
)

left, right = st.columns([1.05, 0.95])
with left:
    st.subheader("Classification Report")
    display = report_df.copy()
    numeric_cols = ["precision", "recall", "f1-score", "support"]
    display[numeric_cols] = display[numeric_cols].round(4)
    st.dataframe(display, hide_index=True, use_container_width=True)

with right:
    st.subheader("Confusion Matrix")
    st.plotly_chart(confusion_matrix_chart(matrix_df), use_container_width=True)

st.subheader("Hasil Modeling Notebook")
model_cols = st.columns(4)
model_cols[0].metric("Best CV Macro F1", f"{meta['cv_macro_f1']:.2%}")
model_cols[1].metric("Best C", str(meta["best_params"]["svc__C"]))
model_cols[2].metric("Best Loss", meta["best_params"]["svc__loss"])
model_cols[3].metric("TF-IDF Features", f"{meta['tfidf']['max_features']:,}")

comparison = pd.DataFrame(
    [
        {"Model": "LinearSVC + TF-IDF", "Training Accuracy": 0.9874, "Testing Accuracy": 0.9710, "Macro F1": 0.5632},
        {"Model": "Tuned LinearSVC + TF-IDF", "Training Accuracy": 0.9914, "Testing Accuracy": 0.9740, "Macro F1": 0.5738},
        {"Model": "Random Forest + TF-IDF", "Training Accuracy": 0.9981, "Testing Accuracy": 0.9749, "Macro F1": 0.3795},
        {"Model": "Logistic Regression + SMOTE", "Training Accuracy": 0.9508, "Testing Accuracy": 0.9217, "Macro F1": 0.4838},
        {"Model": "BiLSTM + Class Weight", "Training Accuracy": 0.9654, "Testing Accuracy": 0.9400, "Macro F1": 0.4826},
    ]
)
st.dataframe(
    comparison.style.format(
        {
            "Training Accuracy": "{:.2%}",
            "Testing Accuracy": "{:.2%}",
            "Macro F1": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption("Model final memakai Tuned LinearSVC karena macro F1 tertinggi di notebook modeling.")



with st.expander("Konfigurasi model terbaik"):
    st.write(
        {
            "model": "Pipeline(Normalizer(norm='l2'), LinearSVC(class_weight='balanced'))",
            "grid_search_best_params": meta["best_params"],
            "tfidf": meta["tfidf"],
            "weighted_f1": round(float(weighted["f1-score"]), 4),
            "catatan": "Confidence pada halaman prediksi berasal dari margin LinearSVC, bukan probabilitas asli.",
        }
    )
