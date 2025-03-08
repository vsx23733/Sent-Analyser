import pandas as pd

def load_data(path_to_data: str) -> pd.DataFrame:
    """
    Load data from the CSV file of Google reviews.

    Parameters:
        - path_to_data (str): Path to the CSV file of Google reviews.

    Returns:
        - DataFrame: Loaded data
    """
    return pd.read_csv(path_to_data, sep=",")


def to_sentiment(rating) -> int:
    """
    Convert rating to sentiment.

    Parameters:
        - rating (int or float): Rating of the review.

    Returns:
        - sentiment_score (int): Sentiment score of the review.
    """
    rating = pd.to_numeric(rating, errors='coerce')  
    if pd.isna(rating):  
        return -1 

    rating = int(rating) 

    if rating <= 2:
        return 0  # Negative sentiment
    elif rating == 3:
        return 1  # Neutral sentiment
    else:
        return 2  # Positive sentiment


def create_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column in the DataFrame with sentiment scores.

    Parameters: 
        - df (pd.DataFrame): DataFrame with reviews.

    Returns:
        - pd.DataFrame: Updated DataFrame
    """
    if 'score' not in df.columns:
        raise ValueError("The input DataFrame does not contain a 'score' column.")

    df['sentiment'] = df['score'].apply(to_sentiment)
    return df
