# CLAUDE.md

## Project Context

Project ini merupakan Capstone Project Machine Learning yang berfokus pada analisis sentimen ulasan pengguna.

Model machine learning telah selesai dikembangkan pada notebook modelling dan model terbaik telah disimpan menggunakan Joblib.

Dashboard akan digunakan untuk:

* Presentasi capstone
* Demonstrasi kemampuan model
* Eksplorasi data
* Prediksi sentimen secara realtime
* Menampilkan performa model kepada dosen dan stakeholder non-teknis

Target pengguna:

* Dosen pembimbing
* Penguji
* Mahasiswa
* Pengguna non-teknis

Prioritas utama:

1. User Experience
2. Visualisasi Data
3. Kecepatan Respon
4. Kemudahan Demonstrasi
5. Kualitas Prediksi

---

# Existing Assets

## Dataset

Dataset utama:

```text
dataset/tokepedia_reviews.csv
```

Dataset telah melalui preprocessing dan digunakan untuk pelatihan model.

---

## Trained Model

Model utama:

```text
modeling/model_svm_best.pkl
```

Vectorizer:

```text
model-result/model_save/tfidf_vectorizer.pkl
```

Model ini adalah source of truth untuk inference.

Jangan mengganti model tanpa instruksi eksplisit.

---

# Technology Stack

Frontend:

* Streamlit

Machine Learning:

* Scikit-learn
* Joblib

Data Processing:

* Pandas
* NumPy

Visualization:

* Plotly
* Matplotlib
* WordCloud

NLP:

* Sastrawi

Deployment:

* Streamlit Community Cloud

---

# Architecture Principles

Seluruh aplikasi harus mengikuti prinsip:

## 1. Separation of Concerns

Business logic tidak boleh berada di halaman Streamlit.

Gunakan struktur:

```text
utils/
├── preprocessing.py
├── prediction.py
├── analytics.py
├── charts.py
```

Halaman Streamlit hanya bertugas:

* menerima input
* memanggil fungsi
* menampilkan output

---

## 2. Reusability

Hindari duplikasi kode.

Jika fungsi digunakan lebih dari satu kali:

* pindahkan ke utils

---

## 3. Maintainability

Kode harus mudah dipahami oleh anggota tim lain.

Prioritaskan:

* nama fungsi jelas
* dokumentasi sederhana
* modularitas

---

# Directory Structure

```text
sentiment-dashboard/

├── app.py

├── data/
│   └── vibesight_ml_ready_v1.0.csv

├── model/
│   ├── model_svm_best.pkl
│   └── tfidf_vectorizer.pkl

├── assets/
│   ├── logo.png
│   ├── banner.png
│   └── confusion_matrix.png

├── pages/
│   ├── Dashboard.py
│   ├── Predict.py
│   ├── Analytics.py
│   ├── Model_Performance.py
│   ├── Batch_Prediction.py
│   └── About.py

├── utils/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── analytics.py
│   └── charts.py

├── requirements.txt

└── README.md
```

---

# Dashboard Requirements

## Dashboard Page

Purpose:

Memberikan gambaran umum terhadap dataset dan project.

Required Components:

* Hero section
* Project summary
* Total dataset
* Sentiment distribution
* KPI cards
* Interactive charts

Preferred Layout:

```text
--------------------------------
Hero Section
--------------------------------

Metric Cards

--------------------------------

Sentiment Distribution

--------------------------------

Dataset Insights

--------------------------------
```

---

## Predict Page

Purpose:

Prediksi sentimen realtime.

Required Features:

* Text input
* Predict button
* Prediction result
* Confidence score
* Processing indicator

Workflow:

```text
Input Text
    ↓
Preprocessing
    ↓
TF-IDF
    ↓
SVM Prediction
    ↓
Display Result
```

---

## Analytics Page

Purpose:

Eksplorasi dataset.

Required Visualizations:

* WordCloud
* Top words
* Sentiment frequency
* Text length distribution
* Positive keyword analysis
* Negative keyword analysis

Preferred Library:

Plotly

Reason:

Interactive and presentation-friendly.

---

## Model Performance Page

Purpose:

Menampilkan kualitas model.

Required Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

Required Visualizations:

* Confusion Matrix
* Classification Report

Important:

Seluruh metrik harus berasal dari hasil training notebook.

Jangan menghitung ulang jika data evaluasi tidak tersedia.

---

## Batch Prediction Page

Purpose:

Prediksi massal menggunakan file CSV.

Required Features:

* Upload CSV
* Preview data
* Predict all rows
* Download result

Expected Output:

```csv
text,prediction
review1,positive
review2,negative
```

---

# Preprocessing Requirements

CRITICAL RULE:

Inference preprocessing harus identik dengan training preprocessing.

Pipeline inference wajib mengikuti notebook modelling.

Jika notebook menggunakan:

* case folding
* cleaning
* stemming
* stopword removal

maka inference wajib menggunakan langkah yang sama.

Perbedaan preprocessing dapat menyebabkan penurunan performa model.

---

# Performance Requirements

Gunakan caching untuk resource berat.

Model loading:

```python
@st.cache_resource
```

Dataset loading:

```python
@st.cache_data
```

Target:

* startup cepat
* minim reload
* pengalaman pengguna lebih baik

---

# UI Guidelines

Theme:

Modern AI Dashboard

Characteristics:

* dark mode
* professional
* clean
* minimal

Avoid:

* excessive colors
* cluttered layouts
* oversized tables
* default Streamlit appearance

---

# Error Handling

Seluruh halaman harus memiliki:

* try-except handling
* informative error messages
* graceful degradation

Aplikasi tidak boleh crash ketika:

* model gagal dimuat
* file CSV tidak valid
* input kosong

---

# Development Philosophy

Saat mengembangkan fitur baru:

1. Prioritaskan pengalaman pengguna.
2. Prioritaskan keterbacaan kode.
3. Hindari overengineering.
4. Fokus pada kebutuhan capstone.
5. Jangan menambahkan teknologi yang tidak memberi nilai nyata.

Jika ada dua solusi:

* pilih solusi yang lebih sederhana
* pilih solusi yang lebih mudah dipresentasikan
* pilih solusi yang lebih mudah dipelihara

Tujuan project ini bukan menunjukkan sebanyak mungkin library.

Tujuannya adalah menunjukkan bahwa model machine learning dapat digunakan secara nyata melalui dashboard yang profesional, stabil, dan mudah digunakan.
