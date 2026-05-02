# Laporan Data Engineering Pipeline — VibeSight

**Proyek:** AI-based Sentiment Analytics for Tokopedia Product Reviews  
**Data Engineer:** Nurhayati Kusumaning Ayu (APC617D6X0220)  
**Versi:** v1.0 | **Tanggal:** 02 May 2026  
**Sumber Data:** Kaggle — Tokopedia Product Reviews 2025  

---

## Ringkasan Eksekutif

Pipeline ini memproses dataset ulasan produk Tokopedia dari kondisi mentah  
hingga siap digunakan untuk pelatihan model sentimen. Total 24 tahap  
pemrosesan diterapkan secara berurutan.

---

## Statistik Dataset

| Metrik | Nilai |
|---|---|
| Total baris (raw) | 65,543 |
| Total baris (clean) | 65,273 |
| Baris dihapus | 270 |
| Total kolom (clean) | 28 |
| Data Training | 52,218 |
| Data Testing | 13,055 |

---

## Distribusi Label Sentimen

| Label | Jumlah | Persentase |
|---|---|---|
| Positif | 63,677 | 97.6% |
| Netral  | 801 | 1.2% |
| Negatif | 795 | 1.2% |

---

## Tahapan Pipeline yang Diterapkan

1. **Data Ingestion** — Load CSV ke DataFrame
2. **Initial Assessment** — Cek shape, kolom, missing values
3. **Data Understanding** — Hapus kolom tidak relevan (product_url, product_variant)
4. **Duplicate Handling** — Hapus duplikat berdasarkan review_id
5. **Missing Values** — Drop baris kritis, imputasi kolom non-kritis
6. **Data Type Fix** — Konversi review_date ke datetime, ID ke string
7. **Format Standardization** — Lowercase, strip whitespace
8. **Value Validation** — Rating 1–5, label valid, teks ≥4 karakter
9. **Outlier Handling** — Flag outlier harga (IQR method)
10. **Text Cleaning** — Hapus URL, emoji, simbol, angka
11. **Text Normalization** — Kamus 63 entri slang→baku
12. **Stopword Removal** — 778 stopword (NLTK + kustom)
13. **Stemming** — PySastrawi Bahasa Indonesia
14. **Feature Engineering** — 8 fitur baru (panjang teks, tahun, kategori harga, dll)
15. **Feature Validation** — Visualisasi distribusi 6 fitur
16. **Encoding** — sentiment_encoded, price_category_encoded, rating_group_encoded
17. **Data Leakage Check** — Verifikasi korelasi fitur vs target
18. **Train-Test Split** — 80:20 dengan stratified sampling
19. **Imbalance Check** — Dokumentasi dan rekomendasi untuk tim ML
20. **Anomaly Detection** — Flag 4 jenis anomali
21. **Final Quality Check** — Semua kolom kritis bebas null dan duplikat
22. **Data Versioning** — Metadata JSON dengan timestamp
23. **Documentation** — Laporan ini
24. **Final Export** — 3 file CSV output

---

## Catatan Penting untuk Tim ML Engineer

- **Fitur Utama NLP**: Gunakan kolom `review_text_stemmed` sebagai input teks
- **Target**: Gunakan kolom `sentiment_encoded` (0=negatif, 1=netral, 2=positif)
- **Imbalance**: Dataset sangat imbalanced — terapkan class_weight atau SMOTE
- **Evaluasi**: Gunakan F1-Score macro, bukan hanya Accuracy
- **Data split**: Gunakan `X_train`/`X_test` yang sudah disediakan (stratified)
