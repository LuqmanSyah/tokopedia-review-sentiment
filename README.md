# Tokopedia Product Review Sentiment Analysis

MVP machine learning untuk klasifikasi sentimen ulasan produk Tokopedia berbahasa Indonesia menjadi tiga kelas:

- Positif
- Netral
- Negatif

Proyek ini menggunakan pipeline NLP tradisional: text preprocessing, TF-IDF, model scikit-learn, evaluasi model, penyimpanan artefak, dan prediksi dari terminal. Scope MVP tidak mencakup deep learning, IndoBERT, API, dashboard, frontend, database, atau deployment.

## Ringkasan Hasil

Model terbaik pada eksperimen MVP terbaru menggunakan preprocessing sentiment-aware dan partial balancing `--max-majority-ratio 3`:

| Metric | Nilai |
| --- | ---: |
| Best model | Multinomial Naive Bayes |
| Accuracy | 0.9652 |
| Precision macro | 0.5231 |
| Recall macro | 0.5317 |
| F1 macro | 0.5272 |

Perbandingan model:

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| --- | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 0.9652 | 0.5231 | 0.5317 | 0.5272 |
| Linear SVM | 0.8997 | 0.4219 | 0.6306 | 0.4575 |
| Logistic Regression | 0.8696 | 0.4114 | 0.6622 | 0.4404 |

Catatan: accuracy terlihat tinggi karena dataset sangat didominasi review positif. Karena itu model final dipilih berdasarkan **F1 macro**, bukan accuracy.

Angka di atas adalah hasil eksperimen tersimpan terakhir. Jika preprocessing atau opsi balancing diubah, jalankan training ulang dan pakai output terbaru di folder `reports/`.

Eksperimen tambahan binary sentiment (`Positif` vs `Negatif`, tanpa kelas `Netral`) memberi hasil lebih stabil:

| Task | Best Model | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| 3-class sentiment | Multinomial Naive Bayes | 0.9652 | 0.5231 | 0.5317 | 0.5272 |
| Binary sentiment | Multinomial Naive Bayes | 0.9822 | 0.6947 | 0.7774 | 0.7284 |

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

- Decode HTML entity, misalnya `&amp;`
- Lowercase
- Hapus URL
- Hapus angka
- Hapus tanda baca
- Hapus karakter selain huruf dan spasi
- Hapus whitespace berlebih
- Tokenisasi sederhana
- Normalisasi slang umum, misalnya `gak`, `nggak`, `tdk` menjadi `tidak`
- Stopword removal bahasa Indonesia menggunakan Sastrawi
- Pertahankan kata negasi/kontras penting seperti `tidak`, `bukan`, `jangan`, `kurang`, dan `tapi`
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

Model terbaik dari eksperimen MVP terbaru adalah **Multinomial Naive Bayes** dengan partial balancing `--max-majority-ratio 3`.

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
│   ├── train_binary.py
│   ├── tune.py
│   ├── predict.py
│   └── utils.py
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.json
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── error_analysis.csv
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
reports/error_analysis.csv
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

Untuk eksperimen partial balancing, misalnya kelas mayoritas maksimal 3x kelas minoritas:

```bash
py -m src.train --max-majority-ratio 3
```

File `reports/error_analysis.csv` berisi review pada data test yang salah diprediksi oleh model terbaik. File ini dipakai untuk analisis manual kasus `Negatif`/`Netral` yang masih tertukar.

## Eksperimen Binary

Kelas `Netral` pada dataset berisi banyak review campuran, misalnya review dengan rating sedang tetapi teksnya memiliki sinyal negatif atau positif kuat. Untuk pembanding yang lebih stabil, proyek ini menyediakan eksperimen binary sentiment dengan membuang kelas `Netral` dan hanya melatih:

- Positif
- Negatif

Jalankan binary training:

```bash
py -m src.train_binary --max-majority-ratio 3
```

Output binary training:

```text
data/processed/binary_cleaned_reviews.csv
models/binary_model.pkl
models/binary_tfidf_vectorizer.pkl
models/binary_metadata.json
reports/binary_model_comparison.csv
reports/binary_classification_report.txt
reports/binary_confusion_matrix.png
reports/binary_error_analysis.csv
```

Prediksi memakai model binary:

```bash
py -m src.predict --model models/binary_model.pkl --vectorizer models/binary_tfidf_vectorizer.pkl --metadata models/binary_metadata.json "barang rusak dan pengiriman lama"
```

Eksperimen binary digunakan sebagai pembanding kualitas model. Pipeline utama tetap 3 kelas karena tujuan awal proyek adalah klasifikasi `Positif`, `Netral`, dan `Negatif`.

Hasil binary experiment:

| Metric | Nilai |
| --- | ---: |
| Best model | Multinomial Naive Bayes |
| Accuracy | 0.9822 |
| Precision macro | 0.6947 |
| Recall macro | 0.7774 |
| F1 macro | 0.7284 |

Performa per kelas binary:

| Kelas | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Negatif | 0.40 | 0.57 | 0.47 | 157 |
| Positif | 0.99 | 0.99 | 0.99 | 11277 |

Hasil binary lebih stabil karena kelas `Netral` yang ambigu dikeluarkan. Namun kelas `Negatif` tetap jauh lebih sedikit daripada `Positif`, sehingga precision negatif masih rendah.

## Tuning

Jalankan tuning ringan dengan validation split:

```bash
py -m src.tune --max-majority-ratio 3
```

Output tuning:

```text
data/processed/tuning_cleaned_reviews.csv
reports/tuning_results.csv
reports/tuned_classification_report.txt
reports/tuned_confusion_matrix.png
reports/tuned_error_analysis.csv
models/tuned_model.pkl
models/tuned_tfidf_vectorizer.pkl
models/tuned_metadata.json
```

Script tuning memilih hyperparameter berdasarkan validation F1 macro, lalu melakukan evaluasi final sekali pada test set. Artefak tuning disimpan sebagai `tuned_*` agar tidak menimpa model final MVP di `models/best_model.pkl`.

Untuk smoke test tuning cepat:

```bash
py -m src.tune --sample-size 5000 --limit-combinations 4
```

## Testing

Jalankan test dasar dari root repo:

```bash
py -m unittest discover -s tests
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
| Binary sentiment tanpa kelas `Netral` | Multinomial Naive Bayes | 0.7284 | Pembanding praktis Positif vs Negatif |
| Preprocessing sentiment-aware + partial balancing `--max-majority-ratio 3` | Multinomial Naive Bayes | 0.5272 | Hasil terbaik sementara |
| Baseline awal dengan stemming | Linear SVM | 0.5087 | Kuat, tetapi kalah dari partial balancing ratio 3 |
| Tanpa stemming | Linear SVM | 0.4830 | Lebih rendah dari baseline |
| Partial balancing `--max-majority-ratio 5` | Multinomial Naive Bayes | 0.4711 | Terlalu condong kembali ke kelas mayoritas |
| Full undersampling `--balance` | Multinomial Naive Bayes | 0.3912 | Terlalu banyak data positif dibuang |

Kesimpulan eksperimen:

- Stemming membantu performa pada dataset ini.
- Menjaga kata negasi dan normalisasi slang sederhana membantu pipeline sentiment.
- Partial balancing ratio 3 lebih baik daripada baseline awal dan ratio 5.
- Full undersampling belum cocok karena terlalu banyak data positif dibuang.
- Model final MVP sementara menggunakan Multinomial Naive Bayes dengan `--max-majority-ratio 3`.
- Eksperimen binary jauh lebih stabil daripada 3-class, tetapi diposisikan sebagai pembanding karena objective utama tetap 3 kelas.

## Notebook

Notebook di folder `notebooks/` disediakan untuk eksplorasi dan dokumentasi:

- `01_eda.ipynb`: membaca dataset, melihat struktur kolom, missing value, duplikasi, dan distribusi label/rating.
- `02_preprocessing.ipynb`: mencoba fungsi preprocessing dari `src/preprocessing.py`.
- `03_model_training.ipynb`: menjalankan training MVP melalui modul `src.train`.

## Batasan

Beberapa batasan MVP:

- Dataset sangat tidak seimbang dan didominasi kelas Positif.
- Performa kelas Negatif dan Netral masih lebih rendah dibanding kelas Positif.
- Error analysis menunjukkan kelas Netral ambigu karena banyak review rating/label Netral memiliki teks campuran atau polaritas kuat.
- Normalisasi slang/kata informal Indonesia masih terbatas pada daftar awal.
- Belum ada tuning hyperparameter mendalam.
- Belum ada model transformer seperti IndoBERT.

## Pengembangan Lanjutan

Prioritas improvement berikutnya:

1. Perluas kamus normalisasi slang Indonesia berdasarkan `reports/error_analysis.csv`.
2. Tambahkan dukungan `--experiment-name` atau `--output-dir` agar hasil eksperimen tidak saling overwrite.
3. Bandingkan hasil 3-class dan binary sentiment dari sisi use case.
4. Tuning parameter Linear SVM dan Logistic Regression.
5. Coba dataset pembanding untuk menguji generalisasi model.
