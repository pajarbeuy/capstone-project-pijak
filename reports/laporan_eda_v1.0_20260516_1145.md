# Laporan EDA — Zenlytics
**Tanggal:** 16 May 2026, 11:45  


## Ringkasan Dataset
- Total baris: 65,273
- Total kolom: 16
- Train set: 52,218 baris | Test set: 13,055 baris

## Distribusi Kelas
| Kelas    | Jumlah | Persentase |
|----------|--------|------------|
| positive | 63,677 | 97.6% |
| neutral  | 801  | 1.2% |
| negative | 795  | 1.2% |

**Rasio Imbalance: 80.1:1**

## Temuan Utama
1. Class imbalance sangat ekstrem (80:1) — strategi wajib: class_weight + SMOTE
2. Rating berkorelasi sempurna dengan sentiment — hindari sebagai fitur utama
3. Ulasan negatif cenderung lebih panjang dari positif
4. 14 baris review_text_stemmed null — handle dengan fillna('')

## Grafik yang Dihasilkan
- eda_01_class_imbalance.png
- eda_02_distribusi_panjang_teks.png
- eda_03_rating_dan_sold.png
- eda_04_top_kata_per_kelas.png
- eda_05_wordcloud.png
- eda_06_kategori_produk.png
- eda_07_kategori_harga.png
- eda_08_analisis_temporal.png
- eda_09_korelasi_fitur.png
