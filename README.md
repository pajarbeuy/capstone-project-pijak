# VibeSight — AI-based Sentiment Analytics for Tokopedia Product Reviews

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
        capstone-project-pijak/
        ├── data/
        │   ├── raw/              # ⚠️ tidak diupload — lihat link dataset di bawah
        │   └── processed/        # hasil pipeline Data Engineering
        │       ├── vibesight_ml_ready_v1.0.csv
        │       ├── vibesight_train_v1.0.csv
        │       ├── vibesight_test_v1.0.csv
        │       └── metadata_v1.0_*.json
        ├── notebooks/
        │   └── 01_data_engineering_pipeline.ipynb
        ├── reports/
        │   ├── feature_distribution.png
        │   └── laporan_data_engineering_v1.0.md
        ├── src/
        └── README.md

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
pip install pandas numpy matplotlib seaborn scikit-learn PySastrawi nltk imbalanced-learn
```

---

## 🚀 Cara Menjalankan Pipeline Data Engineering

1. Download dataset dari link Google Drive di atas
2. Simpan di `data/raw/tokopedia_product_reviews_2025.csv`
3. Buka `notebooks/01_data_engineering_pipeline.ipynb`
4. Jalankan semua cell dari atas ke bawah secara berurutan
5. Output otomatis tersimpan di `data/processed/`

---

## 📋 Output Pipeline

| File | Deskripsi |
|---|---|
| `vibesight_ml_ready_v1.0.csv` | Dataset siap modeling (kolom terpilih) |
| `vibesight_train_v1.0.csv` | Data latih — 52.218 baris (80%) |
| `vibesight_test_v1.0.csv` | Data uji — 13.055 baris (20%) |
| `metadata_v1.0_*.json` | Rekam jejak pipeline & versioning |