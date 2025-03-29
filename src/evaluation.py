import torch
from sklearn.metrics import accuracy_score, classification_report
import json
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentEvaluator:
    def __init__(self, model_path, device=None):
        """
        Initialize the sentiment evaluator with a trained model.
        
        Args:
            model_path (str): Path to the saved model weights
            device (str): Device to run evaluation on ('cuda', 'mps', or 'cpu')
        """
        # Set up device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
        
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, test_texts, test_labels, max_length=128):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_texts (list): List of input texts
            test_labels (list): List of true sentiment labels (0: Negative, 1: Neutral, 2: Positive)
            max_length (int): Maximum length of the input sequence
            
        Returns:
            dict: Evaluation metrics including accuracy and classification report
        """
        all_preds = []
        all_labels = []
        
        # Process each text and predict the sentiment
        for text in test_texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                
            all_preds.append(prediction)
            all_labels.append(test_labels)
        
        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

def save_evaluation_results(results, output_file='output.json'):
    """
    Save the evaluation results to a JSON file.
    
    Args:
        results (dict): Evaluation results
        output_file (str): Path to the output JSON file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")

def main():
    # Define model path and test data
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pt')
    
    # Example test dataset
    test_texts = [
        "I love this product!",
        "It's okay, not the best.",
        "I hate this, it's terrible."
    ]
    test_labels = [2, 1, 0]  # Corresponding labels (0: Negative, 1: Neutral, 2: Positive)
    
    try:
        evaluator = SentimentEvaluator(model_path=model_path)
        results = evaluator.evaluate(test_texts, test_labels)
        
        # Save the results to a JSON file
        save_evaluation_results(results, 'output.json')
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
