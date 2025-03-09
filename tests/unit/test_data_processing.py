import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from src.data_extraction import load_data, to_sentiment, create_sentiment_column
from src.data_processing import preprocess_reviews_processing, SentimentDataset, prepare_dataloader

# Sample test data
def sample_dataframe():
    return load_data(path_to_data=r".\src\reviews.csv")

# Test BERT tokenization
@patch("src.data_processing.tokenizer")
def test_preprocess_reviews(mock_tokenizer):
    df = create_sentiment_column(sample_dataframe())
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                                   "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])}
    
    input_ids, attention_masks, labels = preprocess_reviews_processing(df, max_length=128)
    
    assert input_ids.shape[0] == 3, "Input IDs should have 3 rows"
    assert labels.shape[0] == len(df), "Labels should have len(df) rows"
    assert input_ids.shape[1] == 3, "Input IDs should have 3 columns"

# Test SentimentDataset class
def test_sentiment_dataset():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    attention_masks = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
    labels = torch.tensor([2, 1, 0])
    dataset = SentimentDataset(input_ids, attention_masks, labels)
    assert len(dataset) == 3
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample

# Test prepare_dataloader function
@patch("src.data_processing.load_data_processing")
@patch("src.data_processing.create_sentiment_column_processing")
@patch("src.data_processing.preprocess_reviews_processing")
def test_prepare_dataloader(mock_preprocess, mock_create_sentiment, mock_load_data):
    df = create_sentiment_column(sample_dataframe())
    mock_load_data.return_value = df
    mock_create_sentiment.return_value = df
    mock_preprocess.return_value = (
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
        torch.tensor([2, 1, 0])
    )
    dataloader = prepare_dataloader(r".\src\reviews.csv", batch_size=2)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    batch = next(iter(dataloader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
