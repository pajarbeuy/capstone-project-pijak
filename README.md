# Zenlytics — AI-based Sentiment Analytics for Tokopedia Product Reviews

**Program:** Pijak × IBM SkillsBuild Capstone Project  
**Tema:** AI for Business Intelligence and Market Insights

---

## 👥 Anggota Tim

| ID Peserta | Nama | Peran |
|---|---|---|
| APC237D6Y0070 | Hilman Fauzi Abdilah | AI Engineer / Project Manager |
| APC246D6Y0164 | Ade Wahyu Warpudin | AI Engineer / ML Engineer |
| APC246D6Y0236 | Ade Suryadi | AI Engineer / ML Engineer |
| APC446D6Y0245 | Pajar | AI Engineer / UI-UX Designer |
| APC617D6X0220 | Nurhayati Kusumaning Ayu | AI Engineer / Data Engineer |

---

## 📁 Struktur Folder

```
capstone-project-pijak/
├── data/
│   ├── raw/                          # ⚠️ tidak diupload — lihat link dataset di bawah
│   │   └── tokopedia_product_reviews_2025.csv
│   └── processed/                    # hasil pipeline Data Engineering
│       ├── Zenlytics_clean_full_v1.0.csv
│       ├── Zenlytics_ml_ready_v1.0.csv
│       ├── Zenlytics_train_v1.0.csv
│       ├── Zenlytics_test_v1.0.csv
│       └── metadata_v1.0_*.json
├── notebooks/
│   ├── 01_data_engineering_pipeline.ipynb
│   └── 01.1_eda_zenlytics.ipynb
├── reports/
│   ├── laporan_data_engineering_v1.0.md
│   ├── feature_distribution.png
│   ├── eda_01_class_imbalance.png
│   ├── eda_02_distribusi_panjang_teks.png
│   ├── eda_03_rating_dan_sold.png
│   ├── eda_04_top_kata_per_kelas.png
│   ├── eda_05_wordcloud.png
│   ├── eda_06_kategori_produk.png
│   ├── eda_07_kategori_harga.png
│   ├── eda_10_kata_trigger_negatif.png
│   ├── eda_11_kategori_vs_negatif.png
│   └── eda_12_panjang_vs_sentimen.png
├── modeling/
├── src/
└── README.md
```

---

## 📊 Dataset

- **Sumber:** [Tokopedia Product Reviews 2025 — Kaggle](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025)
- **Dataset raw (Google Drive):** [tokopedia_product_reviews_2025.csv](https://drive.google.com/drive/folders/109lujdSqYDj1CAxtaU6gAUWcWXZVbkgi?usp=sharing)

> Dataset mentah tidak diupload ke GitHub karena ukurannya 22.9 MB.  
> Unduh dari link di atas, lalu simpan di `data/raw/tokopedia_product_reviews_2025.csv`

---

## ⚙️ Setup Environment

```bash
# Clone repository
git clone https://github.com/hilmanfaujiabdilah/capstone-project-pijak.git
cd capstone-project-pijak

# Install semua library yang diperlukan
pip install pandas numpy matplotlib seaborn scikit-learn PySastrawi nltk imbalanced-learn wordcloud
```

---

## 🚀 Cara Menjalankan

### 1. Pipeline Data Engineering

1. Download dataset dari link Google Drive di atas
2. Simpan di `data/raw/tokopedia_product_reviews_2025.csv`
3. Buka `notebooks/01_data_engineering_pipeline.ipynb`
4. Jalankan semua cell dari atas ke bawah secara berurutan
5. Output otomatis tersimpan di `data/processed/`

### 2. Exploratory Data Analysis (EDA)

1. Pastikan pipeline Data Engineering (langkah 1) sudah selesai dijalankan
2. Buka `notebooks/01.1_eda_zenlytics.ipynb`
3. Jalankan semua cell dari atas ke bawah
4. Grafik hasil EDA otomatis tersimpan di `reports/`

---

## 📋 Output Pipeline Data Engineering

| File | Deskripsi |
|---|---|
| `Zenlytics_clean_full_v1.0.csv` | Dataset lengkap semua kolom (34 MB) |
| `Zenlytics_ml_ready_v1.0.csv` | Dataset siap modeling — kolom terpilih (8.2 MB) |
| `Zenlytics_train_v1.0.csv` | Data latih — 52.218 baris (80%) |
| `Zenlytics_test_v1.0.csv` | Data uji — 13.055 baris (20%) |
| `metadata_v1.0_*.json` | Rekam jejak pipeline & versioning |

---

## 📊 Output EDA

| Grafik | Analisis | Insight Utama |
|---|---|---|
| `eda_01_class_imbalance.png` | Distribusi kelas sentimen | Dataset sangat imbalanced — perlu SMOTE atau class_weight |
| `eda_02_distribusi_panjang_teks.png` | Panjang ulasan | Mayoritas ulasan pendek (< 20 kata) |
| `eda_03_rating_dan_sold.png` | Rating & jumlah terjual | Korelasi rating vs sentimen |
| `eda_04_top_kata_per_kelas.png` | Kata dominan per kelas | Kata kunci pembeda positif/netral/negatif |
| `eda_05_wordcloud.png` | WordCloud per kelas | Visualisasi kata paling sering muncul |
| `eda_06_kategori_produk.png` | Distribusi kategori produk | Kategori dominan dalam dataset |
| `eda_07_kategori_harga.png` | Distribusi kategori harga | Segmentasi harga produk |
| `eda_10_kata_trigger_negatif.png` | Kata trigger negatif | Kata eksklusif di ulasan negatif → insight untuk seller |
| `eda_11_kategori_vs_negatif.png` | Kategori vs komplain | Kategori dengan proporsi negatif tertinggi |
| `eda_12_panjang_vs_sentimen.png` | Panjang vs sentimen | Ulasan negatif rata-rata lebih panjang → lebih mudah dideteksi model |

---

## 💡 Rekomendasi untuk Tim ML Engineer

- **Fitur utama NLP:** Gunakan kolom `review_text_stemmed` sebagai input teks
- **Target:** Gunakan kolom `sentiment_encoded` (0=negatif, 1=netral, 2=positif)
- **Imbalance:** Dataset sangat imbalanced — terapkan `class_weight` atau SMOTE
- **Evaluasi:** Gunakan F1-Score macro, bukan hanya Accuracy
- **Data split:** Gunakan `Zenlytics_train_v1.0.csv` dan `Zenlytics_test_v1.0.csv` yang sudah disediakan (stratified 80:20)
- **Teks pendek:** Pisahkan evaluasi untuk ulasan < 10 kata vs ≥ 10 kata