import unittest
import torch
from src.inference import SentimentPredictor
import os


class TestSentimentPredictor(unittest.TestCase):

    def setUp(self):
        """Set up the predictor with a default model path and device."""
        device = os.getenv("DEVICE", "cpu") 
        self.predictor = SentimentPredictor(device=device)

    def test_model_loading(self):
        """Test if the model loads correctly."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsInstance(self.predictor.model, torch.nn.Module)

    def test_device_assignment(self):
        """Check if the device assignment works as expected."""
        self.assertIn(str(self.predictor.device), ['mps', 'cuda', 'cpu'])

    def test_prediction_format(self):
        """Test if the prediction output format is correct."""
        result = self.predictor.predict("I love this product!")
        self.assertIn('sentiment', result)
        self.assertIn('probabilities', result)
        self.assertIn(result['sentiment'], ['Negative', 'Neutral', 'Positive'])
        self.assertEqual(len(result['probabilities']), 3)

    def test_prediction_validity(self):
        """Test prediction with a simple positive statement."""
        result = self.predictor.predict("The movie was fantastic!")
        self.assertIn(result['sentiment'], ['Negative', 'Neutral', 'Positive'])

    def test_empty_text(self):
        """Test prediction with an empty string."""
        result = self.predictor.predict("")
        self.assertIn(result['sentiment'], ['Negative', 'Neutral', 'Positive'])

    def test_long_text(self):
        """Test prediction with a long input text."""
        long_text = "This is a very long text. " * 100
        result = self.predictor.predict(long_text)
        self.assertIn(result['sentiment'], ['Negative', 'Neutral', 'Positive'])

    def test_invalid_device(self):
        """Test initializing with an invalid device."""
        with self.assertRaises(Exception):
            SentimentPredictor(device="invalid_device")


if __name__ == "__main__":
    unittest.main()
