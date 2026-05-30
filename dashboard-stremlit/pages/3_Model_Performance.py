import streamlit as st
import pandas as pd

from utils.charts import confusion_matrix_chart, model_performance
from utils.data import load_modeling_dataset
from utils.ui import page_hero, setup_page


setup_page("Model Performance", "M")
page_hero(
    "Evaluasi Model SVM",
    "Metrik mengikuti output notebook 02 bagian SVM Data Murni: text_akhir, polarity, split stratified 80:20, TF-IDF, lalu SVC linear.",
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
    "Dataset memakai `fix_tokopedia_reviews.csv`, fitur `text_akhir`, target `polarity`, dan nilai kosong diisi string kosong untuk TF-IDF."
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
model_cols[0].metric("Macro F1 Notebook", f"{meta['macro_f1']:.2%}")
model_cols[1].metric("SVM C", str(meta["best_params"]["svm__C"]))
model_cols[2].metric("Kernel", meta["best_params"]["svm__kernel"])
model_cols[3].metric("TF-IDF Features", f"{meta['tfidf']['max_features']:,}")

comparison = pd.DataFrame(
    [
        {"Model": "SVM Baseline", "Testing Accuracy": 0.9365, "Macro F1": 0.8687, "Recall Negative": 0.86, "Recall Neutral": 0.69},
        {"Model": "SVM Grid Search Tuned", "Testing Accuracy": 0.9378, "Macro F1": 0.8633, "Recall Negative": 0.90, "Recall Neutral": 0.69},
        {"Model": "LSTM", "Testing Accuracy": 0.9300, "Macro F1": 0.8695, "Recall Negative": 0.88, "Recall Neutral": 0.88},
    ]
)
st.dataframe(
    comparison.style.format(
        {
            "Testing Accuracy": "{:.2%}",
            "Macro F1": "{:.2%}",
            "Recall Negative": "{:.2%}",
            "Recall Neutral": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption("Website ini memakai artefak SVM Data Murni dari notebook 02: `modeling/model_save/model_svm.pkl` dan `modeling/tfidf_vectorizer.pkl`.")



with st.expander("Konfigurasi model terbaik"):
    st.write(
        {
            "model": "SVC(kernel='linear', C=1)",
            "params": meta["best_params"],
            "tfidf": meta["tfidf"],
            "weighted_f1": round(float(weighted["f1-score"]), 4),
            "catatan": "Confidence pada halaman prediksi berasal dari margin SVM, bukan probabilitas kalibrasi.",
        }
    )
