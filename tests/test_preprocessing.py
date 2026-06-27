import unittest

from src.preprocessing import clean_text, normalize_slang_tokens


class PreprocessingTest(unittest.TestCase):
    def test_normalize_slang_tokens_expands_common_review_terms(self):
        tokens = normalize_slang_tokens(["brg", "gak", "bgt", "gaada"])

        self.assertEqual(tokens, ["barang", "tidak", "banget", "tidak", "ada"])

    def test_clean_text_preserves_negation_for_sentiment(self):
        cleaned = clean_text("Barang gak sesuai &amp; seller ramah", use_stemming=False)

        self.assertIn("tidak", cleaned.split())
        self.assertIn("sesuai", cleaned.split())
        self.assertNotIn("amp", cleaned.split())

    def test_clean_text_replaces_punctuation_with_word_boundary(self):
        cleaned = clean_text("barang tidak-sesuai", use_stemming=False)

        self.assertIn("tidak", cleaned.split())
        self.assertIn("sesuai", cleaned.split())
        self.assertNotIn("tidaksesuai", cleaned.split())

    def test_clean_text_returns_empty_string_for_missing_value(self):
        self.assertEqual(clean_text(None, use_stemming=False), "")


if __name__ == "__main__":
    unittest.main()
