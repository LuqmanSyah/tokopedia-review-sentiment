"""Text preprocessing helpers for Indonesian Tokopedia reviews."""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Iterable

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return Indonesian stopwords from Sastrawi."""
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())


@lru_cache(maxsize=1)
def get_stemmer():
    """Return a cached Sastrawi stemmer instance."""
    return StemmerFactory().create_stemmer()


def clean_text(text: object, use_stemming: bool = True) -> str:
    """Clean, tokenize, remove stopwords, and optionally stem review text."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(_PUNCT_TRANSLATION)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stopwords = get_stopwords()
    tokens = [token for token in text.split() if token not in stopwords]

    if use_stemming and tokens:
        return get_stemmer().stem(" ".join(tokens))

    return " ".join(tokens)


def preprocess_series(texts: Iterable[object], use_stemming: bool = True) -> pd.Series:
    """Apply review preprocessing to a sequence of texts."""
    return pd.Series(texts, dtype="object").apply(
        lambda value: clean_text(value, use_stemming=use_stemming)
    )
