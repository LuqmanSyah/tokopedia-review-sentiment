import unittest

from src.utils import normalize_sentiment_label, sentiment_from_rating


class UtilsTest(unittest.TestCase):
    def test_normalize_sentiment_label_supports_dataset_labels(self):
        self.assertEqual(normalize_sentiment_label("positive"), "Positif")
        self.assertEqual(normalize_sentiment_label("neutral"), "Netral")
        self.assertEqual(normalize_sentiment_label("negative"), "Negatif")

    def test_normalize_sentiment_label_rejects_unknown_label(self):
        self.assertIsNone(normalize_sentiment_label("mixed"))

    def test_sentiment_from_rating_maps_rating_scale(self):
        self.assertEqual(sentiment_from_rating(1), "Negatif")
        self.assertEqual(sentiment_from_rating(2), "Negatif")
        self.assertEqual(sentiment_from_rating(3), "Netral")
        self.assertEqual(sentiment_from_rating(4), "Positif")
        self.assertEqual(sentiment_from_rating(5), "Positif")


if __name__ == "__main__":
    unittest.main()
