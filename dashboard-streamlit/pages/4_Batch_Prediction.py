import pandas as pd
import streamlit as st

from utils.modeling import predict_many
from utils.ui import page_hero, setup_page


setup_page("Batch Prediction", "📂")
page_hero(
    "Batch Prediction",
    "Upload CSV berisi banyak review untuk diprediksi sekaligus dan unduh hasilnya.",
)

uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded is None:
    st.info("CSV minimal memiliki satu kolom teks, misalnya `review_text`.")
else:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("Preview Data")
        st.dataframe(df.head(10), use_container_width=True)

        text_candidates = [col for col in df.columns if "text" in col.lower() or "review" in col.lower()]
        default_index = df.columns.get_loc(text_candidates[0]) if text_candidates else 0
        text_column = st.selectbox("Kolom teks untuk prediksi", df.columns, index=default_index)

        if st.button("Jalankan Batch Prediction", type="primary"):
            with st.spinner("Memproses seluruh review..."):
                results = predict_many(df[text_column].fillna("").astype(str).tolist())
                result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

            st.success(f"Selesai memprediksi {len(result_df):,} baris.")
            st.dataframe(result_df.head(30), use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download hasil CSV",
                data=csv_bytes,
                file_name="vibesight_batch_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Gagal memproses file: {exc}")
