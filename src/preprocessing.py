"""Text preprocessing helpers for Indonesian Tokopedia reviews."""

from __future__ import annotations

import re
import string
from html import unescape
from functools import lru_cache
from typing import Iterable

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


_PUNCT_TRANSLATION = str.maketrans({punctuation: " " for punctuation in string.punctuation})

SENTIMENT_PRESERVE_WORDS = {
    "belum",
    "bukan",
    "jangan",
    "kurang",
    "namun",
    "sesuai",
    "tak",
    "tapi",
    "tidak",
}

SLANG_NORMALIZATION = {
    "brg": "barang",
    "bgt": "banget",
    "bangettt": "banget",
    "bgtt": "banget",
    "dg": "dengan",
    "dgn": "dengan",
    "ga": "tidak",
    "gada": "tidak ada",
    "gak": "tidak",
    "gaada": "tidak ada",
    "gk": "tidak",
    "ngak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "tp": "tapi",
    "yg": "yang",
}


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return Indonesian stopwords from Sastrawi."""
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    return stopwords - SENTIMENT_PRESERVE_WORDS


@lru_cache(maxsize=1)
def get_stemmer():
    """Return a cached Sastrawi stemmer instance."""
    return StemmerFactory().create_stemmer()


def normalize_slang_tokens(tokens: Iterable[str]) -> list[str]:
    """Normalize common Indonesian informal tokens used in product reviews."""
    normalized_tokens = []
    for token in tokens:
        replacement = SLANG_NORMALIZATION.get(token, token)
        normalized_tokens.extend(replacement.split())
    return normalized_tokens


def clean_text(text: object, use_stemming: bool = True) -> str:
    """Clean, tokenize, remove stopwords, and optionally stem review text."""
    if pd.isna(text):
        return ""

    text = unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(_PUNCT_TRANSLATION)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stopwords = get_stopwords()
    tokens = normalize_slang_tokens(text.split())
    tokens = [token for token in tokens if token not in stopwords]

    if use_stemming and tokens:
        return get_stemmer().stem(" ".join(tokens))

    return " ".join(tokens)


def preprocess_series(texts: Iterable[object], use_stemming: bool = True) -> pd.Series:
    """Apply review preprocessing to a sequence of texts."""
    return pd.Series(texts, dtype="object").apply(
        lambda value: clean_text(value, use_stemming=use_stemming)
    )
