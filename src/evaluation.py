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
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
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
        for i, text in enumerate(test_texts):
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
            all_labels.append(test_labels[i]) 

        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

def save_evaluation_results(results, output_file='metrics.json'):
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
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pt')
    
    test_texts = [
            "I love this product!",
            "It's okay, not the best.",
            "I hate this, it's terrible.",
            "This is the best thing I've ever bought!",
            "Not bad, but could be improved.",
            "I'm really disappointed with this purchase.",
            "I can't live without it, amazing!",
            "This is absolutely awful, would not recommend.",
            "It's pretty good, does the job.",
            "Such a waste of money, don't buy it.",
            "I love everything about this!",
            "It could be better, but it's fine.",
            "Worst purchase ever!",
            "The quality is top-notch, very satisfied.",
            "I didn't like it at all.",
            "Best purchase of the year!",
            "This product doesn't meet my expectations.",
            "It's just average, not great, not terrible.",
            "Highly recommend, works great!",
            "Very disappointed, it broke after one use.",
            "I wish I could get my money back.",
            "The product exceeded my expectations!",
            "It's okay, but not worth the price.",
            "I'm extremely happy with this purchase.",
            "It's just alright, nothing special.",
            "Horrible experience, would never buy again.",
            "So impressed with the performance!",
            "Very poor quality, disappointed.",
            "I like it, but there are better options."
        ]

    test_labels = [
            2,  # Positive
            1,  # Neutral
            0,  # Negative
            2,  # Positive
            1,  # Neutral
            0,  # Negative
            2,  # Positive
            0,  # Negative
            1,  # Neutral
            0,  # Negative
            2,  # Positive
            1,  # Neutral
            0,  # Negative
            2,  # Positive
            1,  # Neutral
            2,  # Positive
            0,  # Negative
            1,  # Neutral
            2,  # Positive
            0,  # Negative
            1,  # Neutral
            2,  # Positive
            0,  # Negative
            1,  # Neutral
            2,  # Positive
            0,  # Negative
            1,  # Neutral
            2,  # Positive
            0,  # Negative
            1   # Neutral
        ]

    
    try:
        evaluator = SentimentEvaluator(model_path=model_path)
        results = evaluator.evaluate(test_texts, test_labels)
        
        save_evaluation_results(results, 'metrics.json')
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
