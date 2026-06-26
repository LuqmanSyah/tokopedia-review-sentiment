# Tokopedia Product Review Sentiment Analysis

Proyek ini membangun MVP machine learning untuk klasifikasi sentimen ulasan produk Tokopedia berbahasa Indonesia menjadi tiga kelas:

- Positif
- Netral
- Negatif

Scope MVP sengaja dibatasi ke pipeline NLP tradisional: preprocessing teks, TF-IDF, tiga model scikit-learn, evaluasi, penyimpanan model terbaik, dan prediksi dari terminal.

## Dataset

Dataset utama:

- Kaggle: `salmanabdu/tokopedia-product-reviews-2025`

Simpan file CSV mentah di `data/raw/`. Default script training memakai:

```text
data/raw/tokopedia_product_reviews_2025.csv
```

Dataset ini sudah memiliki kolom `sentiment_label`. Jika dataset lain hanya memiliki rating, script akan membuat label dengan aturan:

- Rating 1 dan 2: Negatif
- Rating 3: Netral
- Rating 4 dan 5: Positif

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
├── reports/
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

## Training MVP

Jalankan training dari root repo:

```bash
py -m src.train
```

Atau jika memakai command `python`:

```bash
python -m src.train
```

Script training akan:

1. Membaca dataset CSV.
2. Mendeteksi kolom review dan label/rating.
3. Membersihkan teks review.
4. Menyimpan data bersih ke `data/processed/cleaned_reviews.csv`.
5. Membuat fitur TF-IDF.
6. Melatih 3 model:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
7. Memilih model terbaik berdasarkan F1-score macro.
8. Menyimpan artefak model dan laporan evaluasi.

Output training:

```text
models/best_model.pkl
models/tfidf_vectorizer.pkl
models/metadata.json
reports/model_comparison.csv
reports/classification_report.txt
reports/confusion_matrix.png
data/processed/cleaned_reviews.csv
```

Untuk percobaan cepat:

```bash
py -m src.train --sample-size 5000
```

Jika ingin mematikan stemming:

```bash
py -m src.train --no-stemming
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

## Notebook

Notebook di folder `notebooks/` disediakan untuk eksplorasi dan dokumentasi eksperimen:

- `01_eda.ipynb`: membaca dataset, melihat struktur kolom, missing value, duplikasi, dan distribusi label/rating.
- `02_preprocessing.ipynb`: mencoba fungsi preprocessing dari `src/preprocessing.py`.
- `03_model_training.ipynb`: menjalankan training MVP melalui modul `src.train`.

## Catatan

- TF-IDF hanya di-fit pada data training untuk menghindari data leakage.
- Preprocessing prediksi memakai fungsi yang sama dengan training.
- Model terbaik dipilih menggunakan F1-score macro karena data sentimen biasanya tidak seimbang.
- Deep learning, IndoBERT, dashboard, API, frontend, database, dan deployment tidak termasuk scope MVP.
