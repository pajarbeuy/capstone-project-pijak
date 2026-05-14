# CODEX.md

## Project Overview

Project ini adalah dashboard analisis sentimen berbasis machine learning menggunakan Streamlit.

Tujuan utama dashboard:

* Menampilkan insight dataset analisis sentimen
* Melakukan prediksi sentimen secara realtime
* Menampilkan evaluasi performa model
* Menyediakan visualisasi interaktif untuk kebutuhan presentasi capstone

Dashboard HARUS:

* clean
* modern
* responsif
* ringan
* mudah dipahami dosen/non-teknis

Project menggunakan:

* Python
* Streamlit
* Scikit-learn
* TF-IDF
* SVM (model utama)
* Plotly
* Pandas

---

# Tech Stack

## Core

* Python 3.11+
* Streamlit
* Scikit-learn
* Pandas
* Numpy
* Joblib

## Visualization

* Plotly
* Matplotlib
* WordCloud

## NLP

* Sastrawi
* Regex
* NLTK (optional)

---

# Project Structure

```txt
sentiment-dashboard/
│
├── app.py
│
├── model/
│   ├── model_svm_best.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   └── vibesight_ml_ready_v1.0.csv
│
├── assets/
│   ├── logo.png
│   ├── confusion_matrix.png
│   └── banner.png
│
├── utils/
│   ├── preprocessing.py
│   ├── predict.py
│   └── charts.py
│
├── pages/
│   ├── 1_🏠_Dashboard.py
│   ├── 2_🔍_Predict_Sentiment.py
│   ├── 3_📊_Analytics.py
│   ├── 4_🧠_Model_Performance.py
│   └── 5_📂_Batch_Prediction.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
│
└── README.md
```

---

# Core Requirements

## 1. Dashboard Home

Halaman utama HARUS menampilkan:

* Total dataset
* Jumlah sentimen positif
* Jumlah sentimen negatif
* Jumlah sentimen netral
* Distribusi sentimen
* Ringkasan project
* Hero section/title

Gunakan:

* st.metric()
* Plotly charts
* responsive columns

---

## 2. Predict Sentiment

Halaman prediksi HARUS:

* menerima input text
* melakukan preprocessing
* melakukan transform TF-IDF
* menjalankan model SVM
* menampilkan hasil sentimen

Tambahkan:

* confidence score/probability jika memungkinkan
* loading animation/spinner

Contoh flow:

1. User input review
2. Text dibersihkan
3. Text di-vectorize
4. Model memprediksi
5. Result muncul dengan styling modern

---

## 3. Analytics Page

Halaman analytics HARUS menampilkan:

* Wordcloud
* Most common words
* Distribusi panjang text
* Distribusi sentimen
* Top positive words
* Top negative words

Gunakan:

* Plotly
* WordCloud
* Counter/Frequency analysis

---

## 4. Model Performance

Halaman evaluasi HARUS menampilkan:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

Tambahkan penjelasan sederhana tentang arti metric.

Karena banyak orang bisa bikin angka 95% tapi gak ngerti apa artinya. Industri penuh dengan ritual statistik performatif seperti itu.

---

## 5. Batch Prediction

Halaman batch prediction HARUS:

* menerima upload CSV
* memproses banyak data sekaligus
* menghasilkan kolom prediksi
* menyediakan tombol download hasil

Gunakan:

* st.file_uploader()
* st.download_button()

---

# UI/UX Guidelines

## Design Style

Gunakan:

* dark modern theme
* minimalis
* clean spacing
* rounded container feel
* professional AI dashboard look

HINDARI:

* warna terlalu mencolok
* terlalu banyak chart
* layout sempit
* tabel berantakan

---

# Streamlit Configuration

Buat file:

```txt
.streamlit/config.toml
```

Isi:

```toml
[theme]
primaryColor="#00ADB5"
backgroundColor="#0F172A"
secondaryBackgroundColor="#1E293B"
textColor="#F8FAFC"
font="sans serif"
```

---

# Coding Rules

## General Rules

* Code harus modular
* Jangan taruh semua logic di app.py
* Gunakan reusable functions
* Gunakan folder utils/
* Gunakan naming yang konsisten

---

# Preprocessing Rules

SEMUA preprocessing inference HARUS sama dengan preprocessing training notebook.

Pipeline preprocessing:

1. lowercase
2. remove punctuation
3. remove number
4. remove URL
5. stemming
6. stopword removal

Gunakan preprocessing yang identik dengan notebook modelling.

Jika preprocessing berbeda:

* distribusi feature berubah
* hasil prediksi bisa tidak konsisten

Dan manusia lalu menyalahkan model ML, padahal yang berubah pipeline-nya sendiri. Tradisi teknologi modern.

---

# Performance Rules

Gunakan caching:

```python
@st.cache_resource
```

untuk:

* load model
* load vectorizer

Gunakan:

```python
@st.cache_data
```

untuk:

* dataset
* analytics

Karena Streamlit rerun script setiap interaksi. Sedikit klik, seluruh universe Python lahir ulang.

---

# Required Features

## MUST HAVE

* Multipage app
* Sidebar navigation
* Interactive charts
* Real-time prediction
* Responsive layout
* Error handling
* Loading spinner
* Batch prediction

## NICE TO HAVE

* Lottie animation
* Authentication
* Theme switcher
* Export PDF
* Model comparison

---

# Deployment

Deployment target:

* Streamlit Community Cloud

Repository:

* GitHub

Deployment flow:

1. Push project ke GitHub
2. Connect ke Streamlit Cloud
3. Deploy app.py

---

# requirements.txt

```txt
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
wordcloud
joblib
Sastrawi
```

---

# Recommended Libraries

## Visualization

* Plotly preferred over Matplotlib

## NLP

* Sastrawi untuk stemming bahasa Indonesia

## Layout

Gunakan:

* st.columns()
* st.container()
* st.expander()
* st.tabs()

---

# Navigation Structure

Sidebar menu:

```txt
🏠 Dashboard
🔍 Predict Sentiment
📊 Analytics
🧠 Model Performance
📂 Batch Prediction
ℹ️ About Project
```

---

# Expected Final Result

Dashboard harus terasa seperti:

* produk AI modern
* clean
* profesional
* siap presentasi capstone

BUKAN:

* notebook yang dipindahin ke browser
* tampilan default Streamlit polos
* dashboard penuh tabel tanpa storytelling

User harus bisa:

* memahami insight
* mencoba prediksi
* melihat performa model
* memahami value project

tanpa harus membaca notebook 300 baris yang isinya import library sebanyak ritual pemanggilan dewa statistik.
