"""Shared utilities for loading data and preparing sentiment labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REVIEW_COLUMN_CANDIDATES = (
    "review_text",
    "review",
    "text",
    "content",
    "ulasan",
    "komentar",
    "comment",
)

LABEL_COLUMN_CANDIDATES = (
    "sentiment_label",
    "sentiment",
    "label",
    "sentimen",
    "target",
)

RATING_COLUMN_CANDIDATES = (
    "rating",
    "ratings",
    "star",
    "stars",
    "score",
    "nilai",
)

SENTIMENT_ORDER = ["Negatif", "Netral", "Positif"]


def find_first_existing_column(
    columns: pd.Index, candidates: tuple[str, ...], explicit: str | None = None
) -> str | None:
    """Find a dataframe column by explicit name or common candidate names."""
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column '{explicit}' tidak ditemukan di dataset.")
        return explicit

    normalized_columns = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized_columns:
            return normalized_columns[candidate.lower()]
    return None


def normalize_sentiment_label(value: object) -> str | None:
    """Normalize common sentiment labels to Indonesian class names."""
    if pd.isna(value):
        return None

    label = str(value).strip().lower()
    mapping = {
        "positive": "Positif",
        "positif": "Positif",
        "pos": "Positif",
        "1": "Positif",
        "neutral": "Netral",
        "netral": "Netral",
        "neu": "Netral",
        "0": "Netral",
        "negative": "Negatif",
        "negatif": "Negatif",
        "neg": "Negatif",
        "-1": "Negatif",
    }
    return mapping.get(label)


def sentiment_from_rating(value: object) -> str | None:
    """Map ratings to sentiment labels: 1-2 negative, 3 neutral, 4-5 positive."""
    if pd.isna(value):
        return None

    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None

    if rating <= 2:
        return "Negatif"
    if rating == 3:
        return "Netral"
    if rating >= 4:
        return "Positif"
    return None


def load_reviews_dataset(
    data_path: str | Path,
    review_column: str | None = None,
    label_column: str | None = None,
    rating_column: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load reviews and return a dataframe with review_text and sentiment columns."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)
    detected_review_column = find_first_existing_column(
        df.columns, REVIEW_COLUMN_CANDIDATES, review_column
    )
    if detected_review_column is None:
        raise ValueError(
            "Kolom review tidak ditemukan. Gunakan --review-column untuk menentukan kolom teks."
        )

    detected_label_column = find_first_existing_column(
        df.columns, LABEL_COLUMN_CANDIDATES, label_column
    )
    detected_rating_column = find_first_existing_column(
        df.columns, RATING_COLUMN_CANDIDATES, rating_column
    )

    prepared = pd.DataFrame()
    prepared["review_text"] = df[detected_review_column]

    source_column: str
    if detected_label_column is not None:
        prepared["sentiment"] = df[detected_label_column].apply(normalize_sentiment_label)
        source_column = detected_label_column
    elif detected_rating_column is not None:
        prepared["sentiment"] = df[detected_rating_column].apply(sentiment_from_rating)
        source_column = detected_rating_column
    else:
        raise ValueError(
            "Kolom sentiment/rating tidak ditemukan. Gunakan --label-column atau --rating-column."
        )

    prepared = prepared.dropna(subset=["review_text", "sentiment"])
    prepared["review_text"] = prepared["review_text"].astype(str)
    prepared = prepared[prepared["review_text"].str.strip().ne("")]
    prepared = prepared[prepared["sentiment"].isin(SENTIMENT_ORDER)]
    prepared = prepared.drop_duplicates(subset=["review_text", "sentiment"])
    prepared = prepared.reset_index(drop=True)

    if prepared.empty:
        raise ValueError("Tidak ada data valid setelah pembersihan review dan label.")

    return prepared, detected_review_column, source_column
