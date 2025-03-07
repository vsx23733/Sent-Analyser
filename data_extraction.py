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

    
