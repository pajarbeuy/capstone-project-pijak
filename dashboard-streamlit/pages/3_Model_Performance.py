import streamlit as st
import pandas as pd

from utils.charts import confusion_matrix_chart, model_performance
from utils.data import load_modeling_dataset
from utils.ui import page_hero, setup_page


setup_page("Model Performance", "M")
page_hero(
    "Evaluasi Model SVM",
    "Performa model pada data testing"
)

try:
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
        f"Split modeling: {meta['train_size']:,} data training dan {meta['test_size']:,} data testing."
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

    st.subheader("Ringkasan Model")
    
    with st.expander("Konfigurasi model terbaik"):
        st.write({
            "model": "SVM",
            "vectorizer": meta["svm_config"]["vectorizer"],
            "kernel": meta["svm_config"]["kernel"],
            "macro_f1": round(float(meta["macro_f1"]), 4),
        })

except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses evaluasi model: {str(e)}")
    st.info("Pastikan model dan vectorizer tersedia di path yang benar.")
