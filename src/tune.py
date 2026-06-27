"""Tune TF-IDF and baseline model hyperparameters for sentiment classification."""

from __future__ import annotations

import argparse
import json
from itertools import product
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    from src.train import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODELS_DIR,
        DEFAULT_PROCESSED_DIR,
        DEFAULT_REPORTS_DIR,
        balance_training_data,
        evaluate_model,
        preprocess_with_progress,
        sample_data,
        save_confusion_matrix,
        save_error_analysis,
    )
    from src.utils import SENTIMENT_ORDER, load_reviews_dataset
except ModuleNotFoundError:
    from train import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODELS_DIR,
        DEFAULT_PROCESSED_DIR,
        DEFAULT_REPORTS_DIR,
        balance_training_data,
        evaluate_model,
        preprocess_with_progress,
        sample_data,
        save_confusion_matrix,
        save_error_analysis,
    )
    from utils import SENTIMENT_ORDER, load_reviews_dataset


TFIDF_GRID = [
    {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": False,
    },
    {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    },
    {
        "max_features": 10000,
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    },
    {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.95,
        "sublinear_tf": True,
    },
]

MODEL_GRID = [
    ("Multinomial Naive Bayes", {"alpha": alpha})
    for alpha in (0.1, 0.3, 0.5, 1.0)
] + [
    ("Linear SVM", {"C": c_value})
    for c_value in (0.3, 0.5, 1.0, 2.0)
] + [
    ("Logistic Regression", {"C": c_value})
    for c_value in (0.3, 0.5, 1.0, 2.0)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF and model hyperparameters using a validation split."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path dataset CSV.")
    parser.add_argument("--review-column", default=None, help="Nama kolom teks review.")
    parser.add_argument("--label-column", default=None, help="Nama kolom label sentimen.")
    parser.add_argument("--rating-column", default=None, help="Nama kolom rating.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi data test final.")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Proporsi validation dari data train sementara.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Seed split data.")
    parser.add_argument("--sample-size", type=int, default=None, help="Batasi data untuk tuning cepat.")
    parser.add_argument(
        "--max-majority-ratio",
        type=float,
        default=3.0,
        help="Batasi kelas mayoritas di data train tuning maksimal N kali kelas minoritas.",
    )
    parser.add_argument("--no-stemming", action="store_true", help="Matikan stemming Sastrawi.")
    parser.add_argument(
        "--limit-combinations",
        type=int,
        default=None,
        help="Batasi jumlah kombinasi untuk smoke test tuning.",
    )
    return parser.parse_args()


def make_model(model_name: str, params: dict[str, Any], random_state: int):
    if model_name == "Multinomial Naive Bayes":
        return MultinomialNB(**params)
    if model_name == "Linear SVM":
        return LinearSVC(class_weight="balanced", random_state=random_state, **params)
    if model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            **params,
        )
    raise ValueError(f"Model tidak dikenal: {model_name}")


def json_ready_params(params: dict[str, Any]) -> dict[str, Any]:
    serializable = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable


def fit_and_evaluate(
    x_train_text: pd.Series,
    y_train: pd.Series,
    x_eval_text: pd.Series,
    y_eval: pd.Series,
    tfidf_params: dict[str, Any],
    model_name: str,
    model_params: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, object], object, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(**tfidf_params)
    x_train = vectorizer.fit_transform(x_train_text)
    x_eval = vectorizer.transform(x_eval_text)

    model = make_model(model_name, model_params, random_state)
    model.fit(x_train, y_train)
    result = evaluate_model(model, x_eval, y_eval)
    return result, model, vectorizer


def prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    use_stemming = not args.no_stemming
    data, review_column, target_source_column = load_reviews_dataset(
        args.data,
        review_column=args.review_column,
        label_column=args.label_column,
        rating_column=args.rating_column,
    )
    print(f"Loaded {len(data):,} valid reviews.")

    if args.sample_size:
        print(f"Sampling {min(args.sample_size, len(data)):,} reviews for tuning.")
        data = sample_data(data, args.sample_size, args.random_state)

    print(f"Preprocessing text. Stemming: {'on' if use_stemming else 'off'}")
    data["clean_review"] = preprocess_with_progress(
        data["review_text"],
        use_stemming=use_stemming,
    )
    data = data[data["clean_review"].str.strip().ne("")].reset_index(drop=True)
    if data["sentiment"].nunique() < 2:
        raise ValueError("Tuning membutuhkan minimal 2 kelas sentimen.")

    return data, review_column, target_source_column


def main() -> None:
    args = parse_args()
    use_stemming = not args.no_stemming

    if args.max_majority_ratio < 1:
        raise ValueError("--max-majority-ratio harus bernilai minimal 1.")

    models_dir = DEFAULT_MODELS_DIR
    reports_dir = DEFAULT_REPORTS_DIR
    processed_dir = DEFAULT_PROCESSED_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    data, review_column, target_source_column = prepare_data(args)
    data.to_csv(processed_dir / "tuning_cleaned_reviews.csv", index=False)

    stratify = data["sentiment"] if data["sentiment"].value_counts().min() >= 2 else None
    x_trainval_text, x_test_text, y_trainval, y_test = train_test_split(
        data["clean_review"],
        data["sentiment"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    trainval_data = pd.DataFrame({"clean_review": x_trainval_text, "sentiment": y_trainval})
    trainval_stratify = (
        trainval_data["sentiment"]
        if trainval_data["sentiment"].value_counts().min() >= 2
        else None
    )
    x_tune_text, x_val_text, y_tune, y_val = train_test_split(
        x_trainval_text,
        y_trainval,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=trainval_stratify,
    )

    x_tune_balanced, y_tune_balanced = balance_training_data(
        x_tune_text,
        y_tune,
        args.random_state,
        max_majority_ratio=args.max_majority_ratio,
    )

    combinations = list(product(TFIDF_GRID, MODEL_GRID))
    if args.limit_combinations is not None:
        combinations = combinations[: args.limit_combinations]

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_f1 = -1.0

    print(f"Running {len(combinations)} tuning combinations.")
    for index, (tfidf_params, model_config) in enumerate(combinations, start=1):
        model_name, model_params = model_config
        print(f"[{index}/{len(combinations)}] {model_name} {model_params} {tfidf_params}")

        result, _, _ = fit_and_evaluate(
            x_tune_balanced,
            y_tune_balanced,
            x_val_text,
            y_val,
            tfidf_params,
            model_name,
            model_params,
            args.random_state,
        )

        row = {
            "rank": None,
            "model": model_name,
            "model_params": json.dumps(model_params, sort_keys=True),
            "tfidf_params": json.dumps(json_ready_params(tfidf_params), sort_keys=True),
            "validation_accuracy": result["accuracy"],
            "validation_precision_macro": result["precision_macro"],
            "validation_recall_macro": result["recall_macro"],
            "validation_f1_macro": result["f1_macro"],
        }
        rows.append(row)

        if float(result["f1_macro"]) > best_f1:
            best_f1 = float(result["f1_macro"])
            best_row = row

    if best_row is None:
        raise RuntimeError("Tidak ada kombinasi tuning yang berhasil dijalankan.")

    results = pd.DataFrame(rows).sort_values("validation_f1_macro", ascending=False)
    results["rank"] = range(1, len(results) + 1)
    results.to_csv(reports_dir / "tuning_results.csv", index=False)

    best_model_name = str(best_row["model"])
    best_model_params = json.loads(str(best_row["model_params"]))
    best_tfidf_params = json.loads(str(best_row["tfidf_params"]))
    best_tfidf_params["ngram_range"] = tuple(best_tfidf_params["ngram_range"])

    print("Best validation configuration:")
    print(results.head(1).to_string(index=False))

    x_trainval_balanced, y_trainval_balanced = balance_training_data(
        x_trainval_text,
        y_trainval,
        args.random_state,
        max_majority_ratio=args.max_majority_ratio,
    )
    test_result, tuned_model, tuned_vectorizer = fit_and_evaluate(
        x_trainval_balanced,
        y_trainval_balanced,
        x_test_text,
        y_test,
        best_tfidf_params,
        best_model_name,
        best_model_params,
        args.random_state,
    )

    labels = [label for label in SENTIMENT_ORDER if label in set(y_test) | set(test_result["predictions"])]
    report = classification_report(
        y_test,
        test_result["predictions"],
        labels=labels,
        zero_division=0,
    )
    (reports_dir / "tuned_classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        y_test,
        test_result["predictions"],
        reports_dir / "tuned_confusion_matrix.png",
    )
    save_error_analysis(
        data,
        y_test,
        test_result["predictions"],
        reports_dir / "tuned_error_analysis.csv",
    )

    joblib.dump(tuned_model, models_dir / "tuned_model.pkl")
    joblib.dump(tuned_vectorizer, models_dir / "tuned_tfidf_vectorizer.pkl")

    metadata = {
        "best_model": best_model_name,
        "best_model_params": best_model_params,
        "best_validation_f1_macro": best_f1,
        "test_accuracy": test_result["accuracy"],
        "test_precision_macro": test_result["precision_macro"],
        "test_recall_macro": test_result["recall_macro"],
        "test_f1_macro": test_result["f1_macro"],
        "review_column": review_column,
        "target_source_column": target_source_column,
        "use_stemming": use_stemming,
        "max_majority_ratio": args.max_majority_ratio,
        "labels": labels,
        "rows_total": int(len(data)),
        "rows_train_tuning": int(len(x_tune_balanced)),
        "rows_validation": int(len(x_val_text)),
        "rows_train_final": int(len(x_trainval_balanced)),
        "rows_test": int(len(x_test_text)),
        "tfidf": json_ready_params(best_tfidf_params),
    }
    (models_dir / "tuned_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Tuning selesai.")
    print(f"Best validation F1 macro: {best_f1:.4f}")
    print(f"Final test F1 macro: {float(test_result['f1_macro']):.4f}")
    print(f"Tuning results: {reports_dir / 'tuning_results.csv'}")
    print(f"Tuned model: {models_dir / 'tuned_model.pkl'}")


if __name__ == "__main__":
    main()
