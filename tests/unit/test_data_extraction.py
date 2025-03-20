import unittest
import pandas as pd
from io import StringIO
from src.data_extraction import load_data, to_sentiment, create_sentiment_column
import os

class TestReviewExtraction(unittest.TestCase):
    
    def setUp(self):
        """
        Set up sample data for testing.
        """
        file_path = os.path.join(os.path.dirname(__file__), '../../src/reviews.csv')
        self.csv_path = file_path 
        self.df = load_data(self.csv_path)
    
    
    def test_load_data(self):
        """Test if the CSV loads correctly into a DataFrame."""
        df = self.df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 12495)
        self.assertIn('score', df.columns)
        self.assertIn('content', df.columns)
    

    def test_to_sentiment(self):
        """Test sentiment classification based on rating values."""
        self.assertEqual(to_sentiment(5), 2)  # Positive
        self.assertEqual(to_sentiment(4), 2)  # Positive
        self.assertEqual(to_sentiment(3), 1)  # Neutral
        self.assertEqual(to_sentiment(2), 0)  # Negative
        self.assertEqual(to_sentiment(1), 0)  # Negative
        
        # Edge cases
        self.assertEqual(to_sentiment(0), 0)  # Lowest possible score
        self.assertEqual(to_sentiment(6), 2)  # Hypothetical max score
        self.assertEqual(to_sentiment(None), -1)  # Handle missing values
        self.assertEqual(to_sentiment("NaN"), -1)  # Handle NaN strings
        self.assertEqual(to_sentiment("3"), 1)  # Handle string inputs
        self.assertEqual(to_sentiment(3.5), 1)  # Floats rounding down
    

    def test_create_sentiment_column(self):
        """Test if the sentiment column is created correctly."""
        df_with_sentiment = create_sentiment_column(self.df)
        self.assertIn('sentiment', df_with_sentiment.columns)
    
    
    def test_create_sentiment_column_missing_score(self):
        """Test handling of missing 'score' column."""
        df = pd.DataFrame({'review': ['Great!', 'Bad!']})
        with self.assertRaises(ValueError):
            create_sentiment_column(df)
    
if __name__ == '__main__':
    unittest.main()
