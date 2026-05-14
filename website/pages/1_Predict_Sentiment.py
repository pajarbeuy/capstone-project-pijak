import streamlit as st

from utils.modeling import predict_sentiment
from utils.ui import LABEL_COLORS, LABEL_TEXT, page_hero, sentiment_badge, setup_page


setup_page("Predict Sentiment", "🔍")
page_hero(
    "Prediksi Sentimen Realtime",
    "Masukkan review produk, lalu model akan memproses teks dan memprediksi sentimennya.",
)

sample_text = "Barang sesuai, pengiriman cepat, kemasan rapi, seller responsif."
text = st.text_area(
    "Review produk",
    value="",
    placeholder=sample_text,
    height=150,
)

col_submit, col_hint = st.columns([0.3, 0.7])
predict_clicked = col_submit.button("Prediksi", type="primary", use_container_width=True)
col_hint.caption("Pipeline notebook 02: cleaning -> case folding -> slang fixing -> stemming -> tokenizing -> stopword -> TF-IDF + SVM")

if predict_clicked:
    if not text.strip():
        st.warning("Masukkan teks review terlebih dahulu.")
    else:
        try:
            with st.spinner("Memproses review dan menjalankan model..."):
                result = predict_sentiment(text)

            st.subheader("Hasil Prediksi")
            left, right = st.columns([0.35, 0.65])
            with left:
                sentiment_badge(result["label"])
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.caption("Confidence dihitung dari margin SVM, bukan probabilitas kalibrasi.")
            with right:
                label = result["label"]
                color = LABEL_COLORS.get(label, "#94A3B8")
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="small-note">Model membaca review ini sebagai</div>
                        <h2 style="color:{color};margin:.25rem 0 0 0;">{LABEL_TEXT.get(label, label.title())}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Lihat teks setelah preprocessing"):
                st.write(result["processed_text"])
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")
