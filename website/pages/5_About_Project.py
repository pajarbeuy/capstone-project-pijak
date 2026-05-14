import streamlit as st

from utils.data import load_metadata
from utils.ui import page_hero, setup_page


setup_page("About Project", "ℹ️")
metadata = load_metadata()

page_hero(
    "About VibeSight",
    "AI-based Sentiment Analytics for Tokopedia Product Reviews.",
)

st.subheader("Tujuan")
st.markdown(
    """
    Dashboard ini dibuat untuk membantu pembacaan insight sentimen review produk secara cepat,
    mulai dari ringkasan dataset, prediksi realtime, analitik kata, evaluasi model, sampai batch prediction.
    """
)

st.subheader("Pipeline")
st.markdown(
    """
    Data diproses melalui cleaning, normalisasi kata tidak baku, stopword removal, stemming Bahasa Indonesia,
    lalu dipakai sebagai fitur teks `review_text_stemmed`. Model utama memakai TF-IDF dan SVM.
    """
)

st.subheader("Metadata")
st.json(
    {
        "project": metadata.get("project", "VibeSight"),
        "version": metadata.get("versi", "v1.0"),
        "created_by": metadata.get("dibuat_oleh", "Nurhayati Kusumaning Ayu"),
        "rows_clean": metadata.get("total_rows_clean"),
        "train_size": metadata.get("train_size"),
        "test_size": metadata.get("test_size"),
        "label_mapping": metadata.get("label_mapping"),
    }
)
