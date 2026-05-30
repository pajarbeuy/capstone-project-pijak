# Sentiment Dashboard

adalah dashboard analisis sentimen review produk Tokopedia berbasis machine learning. Dashboard ini dibuat dengan Streamlit untuk membantu pengguna membaca insight dataset, mencoba prediksi sentimen secara realtime, melihat performa model, dan menjalankan prediksi massal dari file CSV.

Dashboard ini menggunakan model utama **Tuned LinearSVC + TF-IDF** dengan pipeline preprocessing Bahasa Indonesia yang mengikuti notebook data engineering dan modeling.

## Tujuan Dashboard

- Menampilkan ringkasan dataset review produk.
- Menyediakan prediksi sentimen realtime dari input teks.
- Menampilkan analitik teks dan distribusi sentimen.
- Menjelaskan performa model dengan metrik yang mudah dipahami.
- Mendukung batch prediction dari file CSV.
- Menyediakan tampilan clean, modern, dan siap presentasi capstone.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Matplotlib
- WordCloud
- Sastrawi
- NLTK

## Struktur Folder

```txt
dashboard/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- .streamlit/
|   |-- config.toml
|
|-- model/
|   |-- model_svm_best.pkl
|   |-- tfidf_vectorizer.pkl
|
|-- pages/
|   |-- 1_Predict_Sentiment.py
|   |-- 2_Analytics.py
|   |-- 3_Model_Performance.py
|   |-- 4_Batch_Prediction.py
|   |-- 5_About_Project.py
|
|-- utils/
|   |-- charts.py
|   |-- data.py
|   |-- modeling.py
|   |-- paths.py
|   |-- preprocessing.py
|   |-- ui.py
```

## Data dan Model

Dashboard membaca dataset dari:

```txt
../data/processed/vibesight_ml_ready_v1.0.csv
```

Artefak model berada di:

```txt
dashboard/model/model_svm_best.pkl
dashboard/model/tfidf_vectorizer.pkl
```

Model final mengikuti hasil notebook modeling:

- Model: `LinearSVC`
- Pipeline: `Normalizer(norm='l2') + LinearSVC`
- `class_weight`: `balanced`
- Best parameter: `C=0.1`, `loss='squared_hinge'`
- TF-IDF:
  - `max_features=20000`
  - `ngram_range=(1, 2)`
  - `min_df=2`
  - `max_df=0.95`
  - `sublinear_tf=True`

Label sentimen:

```txt
0 = negative
1 = neutral
2 = positive
```

## Pipeline Preprocessing

Preprocessing inference dibuat agar selaras dengan pipeline data engineering:

1. Lowercase
2. Hapus URL
3. Hapus emoji dan karakter unicode non-standar
4. Hapus mention dan hashtag
5. Hapus angka, tanda baca, dan simbol
6. Normalisasi spasi
7. Normalisasi kata tidak baku/slang
8. Stopword removal Bahasa Indonesia
9. Stemming dengan Sastrawi

File terkait:

```txt
utils/preprocessing.py
```

Catatan penting: input realtime diproses lebih dulu sebelum masuk ke TF-IDF dan model SVM. Ini menjaga pipeline inference tetap konsisten dengan training.

## Fitur Dashboard

## 1. Dashboard Home

File:

```txt
app.py
```

Halaman utama menampilkan ringkasan dataset dan insight umum.

Fitur:

- Total jumlah review.
- Jumlah review positif.
- Jumlah review netral.
- Jumlah review negatif.
- Distribusi sentimen dalam bar chart.
- Proporsi sentimen dalam donut chart.
- Top kategori produk berdasarkan jumlah review.
- Ringkasan project dan metadata pipeline.

Manfaat:

- Memberi gambaran cepat kondisi dataset.
- Membantu audiens non-teknis memahami dominasi kelas sentimen.
- Cocok sebagai halaman pembuka saat presentasi.

## 2. Predict Sentiment

File:

```txt
pages/1_Predict_Sentiment.py
```

Halaman ini digunakan untuk memprediksi sentimen dari satu input review.

Alur kerja:

1. User memasukkan teks review.
2. Teks dibersihkan dan diproses dengan pipeline preprocessing.
3. Teks hasil preprocessing diubah menjadi vektor TF-IDF.
4. Model SVM melakukan prediksi.
5. Dashboard menampilkan label sentimen dan confidence.

Output:

- Label prediksi: positif, netral, atau negatif.
- Confidence score.
- Teks hasil preprocessing dalam expander.

Catatan confidence:

Model memakai `LinearSVC`, sehingga tidak memiliki probabilitas native seperti `predict_proba`. Confidence di dashboard dihitung dari margin `decision_function` yang dikonversi dengan pendekatan softmax. Jadi confidence ini adalah estimasi berbasis margin, bukan probabilitas kalibrasi.

## 3. Analytics

File:

```txt
pages/2_Analytics.py
```

Halaman ini menampilkan eksplorasi data teks dan distribusi sentimen.

Fitur pada tab Overview:

- Distribusi sentimen.
- Histogram panjang review berdasarkan jumlah kata.

Fitur pada tab Frekuensi Kata:

- Most common words.
- Top positive words.
- Top negative words.
- Top neutral words.

Fitur pada tab Wordcloud:

- Wordcloud seluruh data.
- Wordcloud per sentimen: positive, neutral, negative.
- Fallback berupa bar chart frekuensi kata jika library `wordcloud` belum tersedia.

Manfaat:

- Membantu memahami kata yang paling sering muncul.
- Membantu melihat pola bahasa pada masing-masing sentimen.
- Memberi bahan storytelling saat presentasi capstone.

## 4. Model Performance

File:

```txt
pages/3_Model_Performance.py
```

Halaman ini menjelaskan performa model berdasarkan notebook modeling.

Evaluasi mengikuti proses notebook:

1. Load dataset `vibesight_ml_ready_v1.0.csv`.
2. Drop null pada kolom `review_text_stemmed`.
3. Split data dengan:
   - `test_size=0.2`
   - `random_state=42`
   - `stratify=y`
4. Transform teks dengan TF-IDF artefak hasil training.
5. Prediksi memakai model `model_svm_best.pkl`.
6. Hitung classification report dan confusion matrix.

Hasil utama:

```txt
Training size   : 52,208
Testing size    : 13,052
Testing accuracy: 97.40%
Macro F1-score  : 57.38%
```

Interpretasi:

- Accuracy tinggi karena model sangat kuat mengenali kelas positif.
- Macro F1 lebih rendah karena kelas negative dan neutral jauh lebih sedikit.
- Model layak sebagai baseline dashboard, tetapi belum ideal untuk sensitivitas tinggi pada kelas minoritas.

Confusion matrix menunjukkan:

- Kelas positive diprediksi sangat baik.
- Kelas negative dan neutral masih sering tertukar atau terdorong menjadi positive.

Halaman ini juga menampilkan perbandingan model dari notebook:

- LinearSVC + TF-IDF
- Tuned LinearSVC + TF-IDF
- Random Forest + TF-IDF
- Logistic Regression + SMOTE
- BiLSTM + Class Weight

Model final yang dipakai dashboard adalah **Tuned LinearSVC + TF-IDF** karena menghasilkan Macro F1 terbaik di notebook.

## 5. Batch Prediction

File:

```txt
pages/4_Batch_Prediction.py
```

Halaman ini digunakan untuk memprediksi banyak review sekaligus dari CSV.

Alur kerja:

1. User upload file CSV.
2. Dashboard menampilkan preview data.
3. User memilih kolom teks yang akan diprediksi.
4. Dashboard menjalankan preprocessing dan prediksi untuk semua baris.
5. Hasil ditampilkan dalam tabel.
6. User dapat mengunduh hasil prediksi sebagai CSV.

Output tambahan:

- `sentiment_predicted`
- `processed_text`
- `confidence`

Format CSV input:

CSV minimal memiliki satu kolom teks, misalnya:

```csv
review_text
"Barang bagus, pengiriman cepat"
"Kualitas biasa saja"
"Produk rusak dan tidak sesuai"
```

## 6. About Project

File:

```txt
pages/5_About_Project.py
```

Halaman ini berisi ringkasan project, pipeline, dan metadata dataset.

Fitur:

- Tujuan dashboard.
- Ringkasan pipeline machine learning.
- Metadata project, versi dataset, jumlah data clean, ukuran train-test, dan label mapping.

## Komponen Utility

## `utils/paths.py`

Menyimpan path utama agar tidak hardcoded di banyak file.

Isi utama:

- Path dashboard.
- Path root project.
- Path model.
- Path vectorizer.
- Path dataset.
- Path metadata.

## `utils/data.py`

Mengatur loading data dengan caching Streamlit.

Fungsi utama:

- `load_dataset()`
- `load_modeling_dataset()`
- `load_train_test()`
- `load_metadata()`
- `sentiment_summary()`

## `utils/preprocessing.py`

Mengatur seluruh pipeline preprocessing inference.

Fungsi utama:

- `clean_text()`
- `normalize_text()`
- `remove_stopwords()`
- `stem_text()`
- `preprocess_text()`

## `utils/modeling.py`

Mengatur loading model dan prediksi.

Fungsi utama:

- `load_model_assets()`
- `predict_sentiment()`
- `predict_many()`
- `margin_to_confidence()`

## `utils/charts.py`

Mengatur pembuatan chart dan metrik evaluasi.

Fungsi utama:

- `sentiment_distribution_chart()`
- `sentiment_donut_chart()`
- `text_length_histogram()`
- `category_chart()`
- `word_frequencies()`
- `frequency_chart()`
- `build_wordcloud()`
- `model_performance()`
- `confusion_matrix_chart()`

## `utils/ui.py`

Mengatur style, tema visual, label, dan komponen UI reusable.

Fungsi utama:

- `setup_page()`
- `inject_css()`
- `page_hero()`
- `sentiment_badge()`

## Cara Menjalankan Lokal

Masuk ke folder dashboard:

```powershell
cd D:\capstone-project-pijak\dashboard
```

Install dependency:

```powershell
pip install -r requirements.txt
```

Jalankan Streamlit:

```powershell
python -m streamlit run app.py --server.port 8502 --server.headless true
```

Buka browser:

```txt
http://localhost:8502
```

## Login dan Konfigurasi MySQL

Semua halaman website sekarang dilindungi form login. Default lokal:

```txt
username: admin
password: admin123
```

Untuk deployment, ubah kredensial lewat environment variable:

```txt
APP_USERNAME=admin
APP_PASSWORD=password-yang-kuat
```

atau buat file `website/.streamlit/secrets.toml` dari contoh `website/.streamlit/secrets.example.toml`:

```toml
[auth]
username = "admin"
password = "password-yang-kuat"

[mysql]
host = "localhost"
port = 3306
database = "capstone_pijak"
user = "root"
password = ""
```

File konfigurasi MySQL ada di `utils/mysql_config.py`. Import `load_mysql_config()` untuk membaca konfigurasi, atau `get_mysql_connection()` jika ingin langsung membuka koneksi memakai `mysql-connector-python`.

## Cara Deploy ke Streamlit Community Cloud

1. Push repository ke GitHub.
2. Pastikan file berikut tersedia:
   - `dashboard/app.py`
   - `dashboard/requirements.txt`
   - `dashboard/model/model_svm_best.pkl`
   - `dashboard/model/tfidf_vectorizer.pkl`
3. Buka Streamlit Community Cloud.
4. Connect repository GitHub.
5. Set main file path:

```txt
dashboard/app.py
```

6. Deploy aplikasi.
