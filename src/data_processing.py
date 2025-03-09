import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from src.data_extraction import load_data, to_sentiment, create_sentiment_column

# Load a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_data_processing(path_to_data: str) -> pd.DataFrame:
    """
    Load data from the CSV file of Google reviews.

    Parameters:
        - path_to_data (str): Path to the CSV file of Google reviews.

    Returns:
        - DataFrame: Loaded data
    """
    return load_data()


def to_sentiment_processing(rating) -> int:
    """
    Convert rating to sentiment.

    Parameters:
        - rating (int or float): Rating of the review.

    Returns:
        - sentiment_score (int): Sentiment score of the review.
    """
    return to_sentiment(rating)


def create_sentiment_column_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column in the DataFrame with sentiment scores.

    Parameters: 
        - df (pd.DataFrame): DataFrame with reviews.

    Returns:
        - pd.DataFrame: Updated DataFrame
    """
    return create_sentiment_column()

def preprocess_reviews_processing(df: pd.DataFrame, max_length=128):
    """
    Tokenizes and processes reviews for BERT input.
    """
    encoding = tokenizer(
        df["content"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    labels = torch.tensor(df["sentiment"].tolist())
    return encoding["input_ids"], encoding["attention_mask"], labels

class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

def prepare_dataloader(path_to_data: str, batch_size=16):
    """
    Complete pipeline: Load data, preprocess, and return a DataLoader.
    """
    df = load_data(path_to_data)
    df = create_sentiment_column(df)
    input_ids, attention_masks, labels = preprocess_reviews_processing(df)
    dataset = SentimentDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

