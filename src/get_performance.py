import json 
import os

def retrieve_accuracy_from_report(metric_file_path: str) -> float:
    """
    This function retrieve the accuracy metric from the metrics report

    Parameters:
        - metric_file_path (str): Path to the metrics.json file

    Returns:
        - accurcay (float): Float value of the accuracy retrieved
    """

    with open(metric_file_path, "r") as fb:
        metric_report = json.load(fb)

    accuracy = round(metric_report["accuracy"], 1)
    print("Accuracy retrieved :", accuracy)
    return accuracy

def main():
    """Main function to retrieve the performance fo the model"""

    METRIC_FILE_PATH = os.path.join(os.path.dirname(__file__), '../metrics.json')

    retrieved_accuracy = retrieve_accuracy_from_report(METRIC_FILE_PATH)
    return retrieved_accuracy

main()