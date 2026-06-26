# PROJECT_CONTEXT.md

# Project Context: Tokopedia Review Sentiment Analysis

## 1. Identitas Proyek

**Nama Repo:** `tokopedia-review-sentiment`

**Nama Proyek:** Tokopedia Product Review Sentiment Analysis

**Jenis Proyek:** Proyek pribadi / portfolio machine learning

**Topik:** Analisis sentimen ulasan produk Tokopedia berbahasa Indonesia

**Tujuan Utama:**
Membangun pipeline machine learning sederhana untuk mengklasifikasikan ulasan produk Tokopedia menjadi sentimen:

* Positif
* Netral
* Negatif

Proyek ini difokuskan untuk memahami alur dasar Natural Language Processing, mulai dari pengambilan dataset, preprocessing teks, ekstraksi fitur menggunakan TF-IDF, pelatihan beberapa model machine learning tradisional, evaluasi model, sampai membuat fungsi prediksi sederhana.

---

## 2. Dataset

Dataset utama yang digunakan:

**Tokopedia Product Reviews 2025**
Kaggle dataset: `salmanabdu/tokopedia-product-reviews-2025`

Dataset ini dipilih karena lebih baru, lebih besar, dan lebih relevan untuk proyek analisis sentimen ulasan e-commerce berbahasa Indonesia.

Dataset alternatif atau pembanding:

**Tokopedia Product Reviews 2019**
Kaggle dataset: `farhan999/tokopedia-product-reviews`

Dataset kedua tidak digunakan di awal agar scope proyek tetap sederhana. Dataset tersebut bisa digunakan di tahap lanjutan untuk pengujian generalisasi model.

---

## 3. Scope Proyek

### Masuk Scope

Proyek ini mencakup:

1. Membaca dataset ulasan produk Tokopedia.
2. Melakukan eksplorasi data sederhana.
3. Membersihkan teks ulasan.
4. Membuat label sentimen jika dataset hanya memiliki rating.
5. Mengubah teks menjadi fitur numerik menggunakan TF-IDF.
6. Melatih 3 model machine learning tradisional:

   * Multinomial Naive Bayes
   * Logistic Regression
   * Linear SVM / LinearSVC
7. Mengevaluasi model menggunakan:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
8. Memilih model terbaik berdasarkan F1-score.
9. Menyimpan model terbaik dalam format `.pkl`.
10. Membuat script prediksi sentimen untuk input teks baru.
11. Membuat dokumentasi proyek di README.

### Di Luar Scope Awal

Hal berikut tidak dikerjakan pada versi awal:

* Deep learning
* IndoBERT
* Transformer model
* Web dashboard
* Deployment
* FastAPI backend
* Frontend React/Next.js
* Database
* Realtime prediction service

Fitur tersebut bisa dikembangkan setelah MVP model selesai.

---

## 4. Tujuan MVP

MVP proyek dianggap selesai jika sudah memiliki:

1. Notebook eksplorasi dan training model.
2. Pipeline preprocessing teks.
3. Perbandingan minimal 3 model.
4. Confusion matrix dan classification report.
5. Model terbaik yang tersimpan.
6. Script sederhana untuk prediksi sentimen dari teks baru.
7. README yang menjelaskan cara menjalankan proyek.

---

## 5. Tech Stack

Bahasa utama:

* Python

Environment:

* Jupyter Notebook
* Google Colab
* Local Python virtual environment

Library utama:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* Sastrawi
* joblib
* kagglehub atau kaggle API

Catatan:

* Gunakan `scikit-learn` untuk TF-IDF, training model, evaluasi, dan splitting data.
* Gunakan `Sastrawi` untuk stemming bahasa Indonesia.
* Gunakan `joblib` untuk menyimpan model dan vectorizer.

---

## 6. Struktur Folder yang Diinginkan

Struktur folder disarankan seperti berikut:

```text
tokopedia-review-sentiment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 7. Aturan Label Sentimen

Jika dataset sudah memiliki kolom label sentimen, gunakan label tersebut.

Jika dataset hanya memiliki rating bintang, gunakan aturan berikut:

* Rating 1 dan 2 = Negatif
* Rating 3 = Netral
* Rating 4 dan 5 = Positif

Pastikan distribusi label diperiksa agar diketahui apakah data seimbang atau tidak.

---

## 8. Preprocessing Teks

Pipeline preprocessing yang digunakan:

1. Mengubah teks menjadi lowercase.
2. Menghapus URL.
3. Menghapus angka.
4. Menghapus tanda baca.
5. Menghapus karakter khusus.
6. Menghapus whitespace berlebih.
7. Tokenization sederhana.
8. Stopword removal bahasa Indonesia.
9. Stemming menggunakan Sastrawi.

Catatan penting:

* Simpan fungsi preprocessing dalam file `src/preprocessing.py`.
* Pastikan preprocessing yang digunakan saat training sama dengan preprocessing saat prediksi.
* Hindari data leakage. TF-IDF hanya boleh di-fit pada data training, bukan seluruh dataset.

---

## 9. Ekstraksi Fitur

Gunakan TF-IDF dengan `TfidfVectorizer` dari scikit-learn.

Konfigurasi awal yang disarankan:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
```

Konfigurasi ini boleh diubah jika hasil model kurang baik.

---

## 10. Model yang Digunakan

Gunakan 3 model utama:

### 1. Multinomial Naive Bayes

Digunakan sebagai baseline karena cepat, ringan, dan sering cocok untuk klasifikasi teks.

### 2. Logistic Regression

Digunakan sebagai model pembanding yang stabil dan cocok untuk klasifikasi multiclass.

### 3. Linear SVM / LinearSVC

Digunakan karena sering memberikan performa kuat pada klasifikasi teks berbasis TF-IDF.

---

## 11. Evaluasi Model

Evaluasi menggunakan:

* Accuracy
* Precision macro
* Recall macro
* F1-score macro
* Classification report
* Confusion matrix

Metric utama untuk memilih model terbaik:

**F1-score macro**

Alasannya, data sentimen biasanya tidak seimbang. F1-score macro lebih adil untuk membandingkan performa antar kelas positif, netral, dan negatif.

---

## 12. Eksperimen Minimal

Minimal lakukan eksperimen berikut:

1. TF-IDF + Naive Bayes
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear SVM

Jika memungkinkan, lakukan eksperimen tambahan:

1. Dengan stemming
2. Tanpa stemming

Tujuannya untuk melihat apakah stemming benar-benar meningkatkan performa model.

---

## 13. Output yang Harus Dihasilkan

Output akhir proyek:

1. `best_model.pkl`
2. `tfidf_vectorizer.pkl`
3. `model_comparison.csv`
4. `confusion_matrix.png`
5. `predict.py`
6. README lengkap

Contoh penggunaan akhir:

```bash
python src/predict.py "barang bagus, pengiriman cepat, seller ramah"
```

Contoh output:

```text
Input Review: barang bagus, pengiriman cepat, seller ramah
Predicted Sentiment: Positif
```

---

## 14. Prinsip Pengerjaan

AI Agent harus mengikuti prinsip berikut:

1. Kerjakan bertahap, jangan langsung membuat sistem terlalu kompleks.
2. Prioritaskan pipeline ML yang rapi dan mudah dipahami.
3. Jangan menggunakan deep learning pada versi awal.
4. Jangan membuat web app sebelum model MVP selesai.
5. Gunakan struktur folder yang bersih.
6. Tulis kode yang mudah dibaca.
7. Tambahkan komentar secukupnya pada bagian penting.
8. Hindari over-engineering.
9. Pastikan setiap script bisa dijalankan.
10. Dokumentasikan cara penggunaan di README.

---

## 15. Roadmap Pengembangan

### Phase 1 — MVP Machine Learning

* Load dataset
* EDA sederhana
* Preprocessing teks
* Labeling sentimen
* TF-IDF
* Training 3 model
* Evaluasi model
* Simpan model terbaik
* Buat script prediksi

### Phase 2 — Improvement

* Tuning hyperparameter
* Bandingkan stemming vs tanpa stemming
* Coba dataset 2019 sebagai data pembanding
* Tambahkan visualisasi distribusi sentimen
* Tambahkan word frequency per sentiment

### Phase 3 — Productization

* Buat API menggunakan FastAPI
* Buat frontend sederhana dengan React atau Next.js
* Fitur input review manual
* Fitur upload CSV
* Dashboard hasil analisis sentimen

### Phase 4 — Advanced NLP

* Coba IndoBERT
* Bandingkan TF-IDF model vs Transformer model
* Fine-tuning model
* Deployment model

---

## 16. Definisi Selesai

Proyek versi awal dianggap selesai jika:

* Dataset berhasil dibaca.
* Data berhasil dibersihkan.
* Label sentimen tersedia.
* Minimal 3 model berhasil dilatih.
* Hasil evaluasi tersimpan.
* Model terbaik berhasil disimpan.
* Script prediksi berhasil berjalan.
* README menjelaskan project, cara instalasi, cara training, dan cara prediksi.
