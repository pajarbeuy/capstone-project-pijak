# 🏆 Model Evaluation & Selection — Capstone Project PIJAK

Proyek ini telah melatih dan membandingkan tiga skenario pemodelan untuk mengklasifikasikan sentimen ulasan pelanggan (**Negatif**, **Netral**, **Positif**). Karena dataset memiliki ketidakseimbangan kelas (*imbalanced data*), evaluasi utama didasarkan pada metrik **Macro F1-Score** dan **Recall pada kelas minoritas**, bukan sekadar Akurasi.

---

## 📊 Perbandingan Performa Model

| Model | Akurasi Testing | Macro F1-Score | Recall (Negatif) | Recall (Netral) |
|---|---|---|---|---|
| SVM (Baseline) | 93.65% | 0.8687 | 0.86 | 0.69 |
| SVM (Grid Search Tuned) | **93.78%** *(Tertinggi)* | 0.8633 | 0.90 | 0.69 |
| LSTM (Deep Learning) | 93.00% | **0.8695** *(Terbaik)* | 0.88 | **0.88** *(Terbaik)* |

---

## 🥇 Model Terbaik: Long Short-Term Memory (LSTM)

Berdasarkan hasil pengujian di atas, **Model LSTM** terpilih sebagai model terbaik untuk tahap deployment dan ekstraksi Business Intelligence.

### Alasan Pemilihan

1. **Keseimbangan Deteksi (Macro F1 Tertinggi)**
   LSTM mencatat Macro F1-Score tertinggi (0.8695), menunjukkan performa yang paling stabil dan tidak bias terhadap kelas mayoritas (Positif).

2. **Keunggulan Ekstraksi Sentimen Minoritas**
   Berbeda dengan model SVM yang sangat kesulitan mendeteksi kelas Netral (Recall hanya 0.69), model LSTM mampu memahami konteks kalimat dengan jauh lebih baik sehingga mencetak Recall **0.88** untuk kelas Netral dan **0.88** untuk kelas Negatif.

3. **Kontekstualisasi Bahasa**
   Sebagai model *sequence-to-sequence*, LSTM terbukti lebih unggul dalam menangkap makna kata yang berurutan dalam teks ulasan dibandingkan ekstraksi fitur tradisional seperti TF-IDF.

---

## Kesimpulan

Kemampuan model ini dalam mendeteksi **komplain (Negatif)** dan **ulasan biasa (Netral)** dengan presisi dan sensitivitas tinggi menjadikannya alat yang sangat ideal untuk menggali *market insights* dan area perbaikan bisnis.
