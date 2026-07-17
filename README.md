# Analisis Sentimen Ulasan Tokopedia

Notebook analisis sentimen ulasan produk Tokopedia 2025 menggunakan TF-IDF dan Logistic Regression.

## Alur Analisis

1. Mengunduh dataset melalui `kagglehub`.
2. Melakukan exploratory data analysis (EDA).
3. Membersihkan teks dan menghapus stopword bahasa Indonesia dengan Sastrawi.
4. Menyeimbangkan kelas sentimen.
5. Melatih model Logistic Regression.
6. Mengevaluasi model dengan classification report dan confusion matrix.
7. Menampilkan word cloud dan mencoba prediksi ulasan baru.

## Menjalankan Notebook

1. Buka `tokopedia_review_sentimen.ipynb` di Jupyter Notebook, JupyterLab, VS Code, atau Google Colab.
2. Jalankan semua cell secara berurutan.

Dependency dipasang oleh cell pertama notebook:

```text
kagglehub
Sastrawi
wordcloud
```

Dependency analisis lain: `pandas`, `matplotlib`, `seaborn`, dan `scikit-learn`.

## Dataset

[Tokopedia Product Reviews 2025](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025) oleh Salman Abdu.

## File

- `tokopedia_review_sentimen.ipynb`: notebook utama.
