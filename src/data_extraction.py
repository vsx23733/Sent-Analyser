import pandas as pd

def load_data(path_to_data: str) -> pd.DataFrame:
    """
    Load data from the CSV file of google reviews.

    Parameters:
        - path_to_data (str): Path to the CSV file of google reviews.

    Returns:
        - Dataframe
    """
    
    raw_data = pd.read_csv(path_to_data, sep=",")
    return raw_data


def to_sentiment(rating: float) -> int:
    """
    Convert rating to sentiment.

    Parameters:
        - rating (int): Rating of the review.

    Returns:
        - sentiment_score (int): Sentiment score of the review.
    """
    
    rating = int(rating)
    
    # Convert to class
    if rating <= 2:
        sentiment_score = 0
        return sentiment_score
    elif rating == 3:
        sentiment_score = 1
        return sentiment_score
    else:
        sentiment_score = 2
        return sentiment_score
    

def create_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column in the dataframe with sentiment scores.

    Parameters: 
        - df (pd.DataFrame): Dataframe with reviews.

    Returns:
        - df (pd.Dataframe): Updated Dataframe
    """

    df['sentiment'] = df.score.apply(to_sentiment)
    return df

