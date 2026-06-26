"""Predict sentiment for a new Tokopedia review from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

try:
    from src.preprocessing import clean_text
except ModuleNotFoundError:
    from preprocessing import clean_text


DEFAULT_MODEL_PATH = Path("models/best_model.pkl")
DEFAULT_VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
DEFAULT_METADATA_PATH = Path("models/metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment from review text.")
    parser.add_argument("review", nargs="*", help="Teks review yang ingin diprediksi.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path best_model.pkl.")
    parser.add_argument(
        "--vectorizer",
        default=str(DEFAULT_VECTORIZER_PATH),
        help="Path tfidf_vectorizer.pkl.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="Path metadata.json.",
    )
    parser.add_argument("--no-stemming", action="store_true", help="Override untuk mematikan stemming.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = " ".join(args.review).strip()
    if not review:
        raise SystemExit("Masukkan teks review. Contoh: py -m src.predict \"barang bagus\"")

    model_path = Path(args.model)
    vectorizer_path = Path(args.vectorizer)
    metadata_path = Path(args.metadata)

    if not model_path.exists() or not vectorizer_path.exists():
        raise SystemExit("Model belum tersedia. Jalankan training terlebih dahulu: py -m src.train")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    use_stemming = bool(metadata.get("use_stemming", True))
    if args.no_stemming:
        use_stemming = False

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned_review = clean_text(review, use_stemming=use_stemming)
    features = vectorizer.transform([cleaned_review])
    prediction = model.predict(features)[0]

    print(f"Input Review: {review}")
    print(f"Predicted Sentiment: {prediction}")


if __name__ == "__main__":
    main()
