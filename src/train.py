"""Train baseline sentiment models for Tokopedia reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    from src.preprocessing import clean_text
    from src.utils import SENTIMENT_ORDER, load_reviews_dataset
except ModuleNotFoundError:
    from preprocessing import clean_text
    from utils import SENTIMENT_ORDER, load_reviews_dataset


DEFAULT_DATA_PATH = Path("data/raw/tokopedia_product_reviews_2025.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF sentiment classifiers for Tokopedia reviews."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path dataset CSV.")
    parser.add_argument("--review-column", default=None, help="Nama kolom teks review.")
    parser.add_argument("--label-column", default=None, help="Nama kolom label sentimen.")
    parser.add_argument("--rating-column", default=None, help="Nama kolom rating.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi data test.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed split data.")
    parser.add_argument("--max-features", type=int, default=5000, help="Max fitur TF-IDF.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.9, help="Maximum document frequency.")
    parser.add_argument("--sample-size", type=int, default=None, help="Batasi jumlah data untuk percobaan cepat.")
    parser.add_argument("--no-stemming", action="store_true", help="Matikan stemming Sastrawi.")
    return parser.parse_args()


def build_models() -> dict[str, object]:
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }


def evaluate_model(model, x_test, y_test) -> dict[str, object]:
    predictions = model.predict(x_test)
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }


def sample_data(data: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Take a balanced-ish sample by class when a quick experiment is requested."""
    if sample_size >= len(data):
        return data.reset_index(drop=True)

    class_counts = data["sentiment"].value_counts()
    if sample_size < len(class_counts):
        raise ValueError(
            f"--sample-size minimal {len(class_counts)} agar setiap kelas sentimen tetap terwakili."
        )

    sampled_parts = []
    remaining = sample_size

    for index, (sentiment, count) in enumerate(class_counts.items()):
        if index == len(class_counts) - 1:
            take = remaining
        else:
            take = max(1, round(sample_size * count / len(data)))
            take = min(take, remaining - (len(class_counts) - index - 1))

        sampled_parts.append(
            data[data["sentiment"] == sentiment].sample(
                n=min(take, count),
                random_state=random_state,
            )
        )
        remaining -= min(take, count)

    sampled = pd.concat(sampled_parts).sample(frac=1, random_state=random_state)
    return sampled.reset_index(drop=True)


def preprocess_with_progress(
    texts: pd.Series,
    use_stemming: bool,
    progress_every: int = 1000,
) -> pd.Series:
    """Preprocess reviews with simple terminal progress for long runs."""
    cleaned_reviews = []
    total = len(texts)

    for index, text in enumerate(texts, start=1):
        cleaned_reviews.append(clean_text(text, use_stemming=use_stemming))
        if index == 1 or index % progress_every == 0 or index == total:
            print(f"Preprocessing review {index:,}/{total:,}...")

    return pd.Series(cleaned_reviews, index=texts.index)


def save_confusion_matrix(y_test, predictions, output_path: Path) -> None:
    labels = [label for label in SENTIMENT_ORDER if label in set(y_test) | set(predictions)]
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    use_stemming = not args.no_stemming

    models_dir = DEFAULT_MODELS_DIR
    reports_dir = DEFAULT_REPORTS_DIR
    processed_dir = DEFAULT_PROCESSED_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data, review_column, target_source_column = load_reviews_dataset(
        args.data,
        review_column=args.review_column,
        label_column=args.label_column,
        rating_column=args.rating_column,
    )

    if args.sample_size:
        data = sample_data(data, args.sample_size, args.random_state)

    data["clean_review"] = preprocess_series(data["review_text"], use_stemming=use_stemming)
    data = data[data["clean_review"].str.strip().ne("")].reset_index(drop=True)
    if data["sentiment"].nunique() < 2:
        raise ValueError("Training membutuhkan minimal 2 kelas sentimen.")

    data.to_csv(processed_dir / "cleaned_reviews.csv", index=False)

    stratify = data["sentiment"] if data["sentiment"].value_counts().min() >= 2 else None
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        data["clean_review"],
        data["sentiment"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    rows: list[dict[str, object]] = []
    trained_models = build_models()
    best_name = None
    best_model = None
    best_predictions = None
    best_f1 = -1.0

    for name, model in trained_models.items():
        model.fit(x_train, y_train)
        result = evaluate_model(model, x_test, y_test)
        rows.append(
            {
                "model": name,
                "accuracy": result["accuracy"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
                "f1_macro": result["f1_macro"],
            }
        )

        if result["f1_macro"] > best_f1:
            best_name = name
            best_model = model
            best_predictions = result["predictions"]
            best_f1 = float(result["f1_macro"])

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    labels = [label for label in SENTIMENT_ORDER if label in set(y_test) | set(best_predictions)]
    report = classification_report(y_test, best_predictions, labels=labels, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(y_test, best_predictions, reports_dir / "confusion_matrix.png")

    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    metadata = {
        "best_model": best_name,
        "best_f1_macro": best_f1,
        "review_column": review_column,
        "target_source_column": target_source_column,
        "use_stemming": use_stemming,
        "labels": labels,
        "rows_total": int(len(data)),
        "rows_train": int(len(x_train_text)),
        "rows_test": int(len(x_test_text)),
        "tfidf": {
            "max_features": args.max_features,
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
        },
    }
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Training selesai.")
    print(f"Best model: {best_name}")
    print(f"Best F1 macro: {best_f1:.4f}")
    print(f"Model comparison: {reports_dir / 'model_comparison.csv'}")
    print(f"Confusion matrix: {reports_dir / 'confusion_matrix.png'}")
    print(f"Saved model: {models_dir / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
