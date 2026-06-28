"""Train binary positive/negative sentiment models for Tokopedia reviews."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    from src.train import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODELS_DIR,
        DEFAULT_PROCESSED_DIR,
        DEFAULT_REPORTS_DIR,
        balance_training_data,
        build_models,
        evaluate_model,
        preprocess_with_progress,
        sample_data,
        save_confusion_matrix,
        save_error_analysis,
    )
    from src.utils import load_reviews_dataset
except ModuleNotFoundError:
    from train import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODELS_DIR,
        DEFAULT_PROCESSED_DIR,
        DEFAULT_REPORTS_DIR,
        balance_training_data,
        build_models,
        evaluate_model,
        preprocess_with_progress,
        sample_data,
        save_confusion_matrix,
        save_error_analysis,
    )
    from utils import load_reviews_dataset


BINARY_LABELS = ["Negatif", "Positif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary positive/negative sentiment classifiers."
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
    parser.add_argument("--sample-size", type=int, default=None, help="Batasi data untuk percobaan cepat.")
    parser.add_argument(
        "--max-majority-ratio",
        type=float,
        default=3.0,
        help="Batasi kelas mayoritas di data train maksimal N kali kelas minoritas.",
    )
    parser.add_argument("--no-stemming", action="store_true", help="Matikan stemming Sastrawi.")
    return parser.parse_args()


def save_binary_confusion_matrix(y_test, predictions) -> None:
    save_confusion_matrix(y_test, predictions, DEFAULT_REPORTS_DIR / "binary_confusion_matrix.png")


def main() -> None:
    args = parse_args()
    use_stemming = not args.no_stemming

    if args.max_majority_ratio < 1:
        raise ValueError("--max-majority-ratio harus bernilai minimal 1.")

    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    data, review_column, target_source_column = load_reviews_dataset(
        args.data,
        review_column=args.review_column,
        label_column=args.label_column,
        rating_column=args.rating_column,
    )
    print(f"Loaded {len(data):,} valid reviews.")

    before_count = len(data)
    data = data[data["sentiment"].isin(BINARY_LABELS)].reset_index(drop=True)
    dropped_neutral = before_count - len(data)
    print(f"Dropped {dropped_neutral:,} neutral reviews for binary experiment.")
    print(f"Binary rows: {len(data):,}")

    if args.sample_size:
        print(f"Sampling {min(args.sample_size, len(data)):,} reviews for binary experiment.")
        data = sample_data(data, args.sample_size, args.random_state)

    print(f"Preprocessing text. Stemming: {'on' if use_stemming else 'off'}")
    data["clean_review"] = preprocess_with_progress(
        data["review_text"],
        use_stemming=use_stemming,
    )
    data = data[data["clean_review"].str.strip().ne("")].reset_index(drop=True)
    if data["sentiment"].nunique() < 2:
        raise ValueError("Binary training membutuhkan kelas Positif dan Negatif.")

    data.to_csv(DEFAULT_PROCESSED_DIR / "binary_cleaned_reviews.csv", index=False)
    print(f"Saved binary cleaned data: {DEFAULT_PROCESSED_DIR / 'binary_cleaned_reviews.csv'}")

    print("Splitting binary train/test data.")
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        data["clean_review"],
        data["sentiment"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["sentiment"],
    )

    x_train_text, y_train = balance_training_data(
        x_train_text,
        y_train,
        args.random_state,
        max_majority_ratio=args.max_majority_ratio,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    print("Fitting binary TF-IDF vectorizer.")
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    rows: list[dict[str, object]] = []
    best_name = None
    best_model = None
    best_predictions = None
    best_f1 = -1.0

    for name, model in build_models().items():
        print(f"Training binary model: {name}")
        model.fit(x_train, y_train)
        result = evaluate_model(model, x_test, y_test)
        print(f"{name} binary F1 macro: {result['f1_macro']:.4f}")
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
    comparison.to_csv(DEFAULT_REPORTS_DIR / "binary_model_comparison.csv", index=False)

    report = classification_report(
        y_test,
        best_predictions,
        labels=BINARY_LABELS,
        zero_division=0,
    )
    (DEFAULT_REPORTS_DIR / "binary_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    save_binary_confusion_matrix(y_test, best_predictions)
    save_error_analysis(
        data,
        y_test,
        best_predictions,
        DEFAULT_REPORTS_DIR / "binary_error_analysis.csv",
    )

    joblib.dump(best_model, DEFAULT_MODELS_DIR / "binary_model.pkl")
    joblib.dump(vectorizer, DEFAULT_MODELS_DIR / "binary_tfidf_vectorizer.pkl")

    metadata = {
        "task": "binary_sentiment",
        "best_model": best_name,
        "best_f1_macro": best_f1,
        "review_column": review_column,
        "target_source_column": target_source_column,
        "use_stemming": use_stemming,
        "max_majority_ratio": args.max_majority_ratio,
        "labels": BINARY_LABELS,
        "rows_total_binary": int(len(data)),
        "rows_dropped_neutral": int(dropped_neutral),
        "rows_train": int(len(x_train_text)),
        "rows_test": int(len(x_test_text)),
        "tfidf": {
            "max_features": args.max_features,
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
        },
    }
    (DEFAULT_MODELS_DIR / "binary_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Binary training selesai.")
    print(f"Best binary model: {best_name}")
    print(f"Best binary F1 macro: {best_f1:.4f}")
    print(f"Binary model comparison: {DEFAULT_REPORTS_DIR / 'binary_model_comparison.csv'}")
    print(f"Binary confusion matrix: {DEFAULT_REPORTS_DIR / 'binary_confusion_matrix.png'}")
    print(f"Binary error analysis: {DEFAULT_REPORTS_DIR / 'binary_error_analysis.csv'}")
    print(f"Saved binary model: {DEFAULT_MODELS_DIR / 'binary_model.pkl'}")


if __name__ == "__main__":
    main()
