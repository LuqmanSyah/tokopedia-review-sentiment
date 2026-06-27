# Tokopedia Product Review Sentiment Analysis

MVP machine learning untuk klasifikasi sentimen ulasan produk Tokopedia berbahasa Indonesia menjadi tiga kelas:

- Positif
- Netral
- Negatif

Proyek ini menggunakan pipeline NLP tradisional: text preprocessing, TF-IDF, model scikit-learn, evaluasi model, penyimpanan artefak, dan prediksi dari terminal. Scope MVP tidak mencakup deep learning, IndoBERT, API, dashboard, frontend, database, atau deployment.

## Ringkasan Hasil

Model terbaik pada eksperimen MVP:

| Metric | Nilai |
| --- | ---: |
| Best model | Linear SVM |
| Accuracy | 0.9583 |
| Precision macro | 0.4913 |
| Recall macro | 0.5327 |
| F1 macro | 0.5087 |

Perbandingan model:

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 0.9583 | 0.4913 | 0.5327 | 0.5087 |
| Logistic Regression | 0.9041 | 0.4141 | 0.5914 | 0.4457 |
| Multinomial Naive Bayes | 0.9729 | 0.6576 | 0.3354 | 0.3329 |

Catatan: accuracy terlihat tinggi karena dataset sangat didominasi review positif. Karena itu model final dipilih berdasarkan **F1 macro**, bukan accuracy.

## Dataset

Dataset utama:

- Kaggle: `salmanabdu/tokopedia-product-reviews-2025`

File dataset disimpan di:

```text
data/raw/tokopedia_product_reviews_2025.csv
```

Dataset memiliki kolom:

- `review_text`
- `rating`
- `sentiment_label`

Jika dataset lain hanya memiliki rating, script akan membuat label sentimen dengan aturan:

- Rating 1 dan 2: Negatif
- Rating 3: Netral
- Rating 4 dan 5: Positif

## Pipeline

Alur utama proyek:

1. Load dataset CSV.
2. Deteksi kolom review dan label/rating.
3. Bersihkan teks review.
4. Split data train/test.
5. Fit TF-IDF hanya pada data train.
6. Latih 3 model baseline.
7. Evaluasi dengan accuracy, precision macro, recall macro, F1 macro, classification report, dan confusion matrix.
8. Pilih model terbaik berdasarkan F1 macro.
9. Simpan model, vectorizer, metadata, dan report.
10. Jalankan prediksi dari terminal.

## Preprocessing

Preprocessing teks dilakukan di `src/preprocessing.py`:

- Lowercase
- Hapus URL
- Hapus angka
- Hapus tanda baca
- Hapus karakter selain huruf dan spasi
- Hapus whitespace berlebih
- Tokenisasi sederhana
- Stopword removal bahasa Indonesia menggunakan Sastrawi
- Stemming bahasa Indonesia menggunakan Sastrawi

Preprocessing yang sama digunakan saat training dan prediksi.

## Fitur Model

Fitur teks dibuat menggunakan TF-IDF:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
```

Fitur yang digunakan:

- Unigram, contoh: `bagus`, `rusak`, `cepat`
- Bigram, contoh: `barang bagus`, `pengiriman cepat`, `tidak sesuai`

## Model

Tiga model baseline yang dilatih:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Model terbaik dari eksperimen final adalah **Linear SVM**.

## Struktur Folder

```text
tokopedia-review-sentiment/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.json
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── model_comparison.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalasi

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training

Jalankan training final dari root repo:

```bash
py -m src.train
```

Atau jika memakai command `python`:

```bash
python -m src.train
```

Output training:

```text
data/processed/cleaned_reviews.csv
models/best_model.pkl
models/tfidf_vectorizer.pkl
models/metadata.json
reports/model_comparison.csv
reports/classification_report.txt
reports/confusion_matrix.png
```

Untuk percobaan cepat:

```bash
py -m src.train --sample-size 5000
```

Untuk eksperimen tanpa stemming:

```bash
py -m src.train --no-stemming
```

Untuk eksperimen balancing dengan random undersampling pada data train:

```bash
py -m src.train --balance
```

## Prediksi

Setelah training selesai:

```bash
py -m src.predict "barang bagus, pengiriman cepat, seller ramah"
```

Contoh output:

```text
Input Review: barang bagus, pengiriman cepat, seller ramah
Predicted Sentiment: Positif
```

Contoh lain:

```bash
py -m src.predict "barang rusak, pengiriman lama, kecewa sekali"
py -m src.predict "produk biasa saja sesuai harga"
```

## Eksperimen MVP

Eksperimen yang sudah dicoba:

| Eksperimen | Best Model | F1 Macro | Catatan |
| --- | --- | ---: | --- |
| Baseline dengan stemming | Linear SVM | 0.5087 | Hasil terbaik MVP |
| Tanpa stemming | Linear SVM | 0.4830 | Lebih rendah dari baseline |
| Full undersampling `--balance` | Multinomial Naive Bayes | 0.3912 | Terlalu banyak data positif dibuang |

Kesimpulan eksperimen:

- Stemming membantu performa pada dataset ini.
- Full undersampling belum cocok untuk hasil final.
- Model final MVP menggunakan baseline dengan stemming dan tanpa `--balance`.

## Notebook

Notebook di folder `notebooks/` disediakan untuk eksplorasi dan dokumentasi:

- `01_eda.ipynb`: membaca dataset, melihat struktur kolom, missing value, duplikasi, dan distribusi label/rating.
- `02_preprocessing.ipynb`: mencoba fungsi preprocessing dari `src/preprocessing.py`.
- `03_model_training.ipynb`: menjalankan training MVP melalui modul `src.train`.

## Batasan

Beberapa batasan MVP:

- Dataset sangat tidak seimbang dan didominasi kelas Positif.
- Performa kelas Negatif dan Netral masih lebih rendah dibanding kelas Positif.
- Normalisasi slang/kata informal Indonesia belum ditambahkan.
- Belum ada tuning hyperparameter mendalam.
- Belum ada model transformer seperti IndoBERT.

## Pengembangan Lanjutan

Prioritas improvement berikutnya:

1. Tambahkan normalisasi slang Indonesia, misalnya `ga`, `gak`, `nggak` menjadi `tidak`.
2. Coba partial balancing, misalnya kelas Positif dibatasi maksimal 3x jumlah kelas minoritas.
3. Tuning parameter Linear SVM dan Logistic Regression.
4. Analisis error untuk review Negatif dan Netral.
5. Coba dataset pembanding untuk menguji generalisasi model.
