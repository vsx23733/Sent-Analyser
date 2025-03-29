import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pt')
    def __init__(self, model_path=model_path, device=None):
        """
        Initialize the sentiment predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model weights
            device (str): Device to run inference on ('cuda', 'mps', or 'cpu')
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
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_length=128):
        """
        Predict sentiment for a given text.
        
        Args:
            text (str): Input text to analyze
            max_length (int): Maximum length of the input sequence
            
        Returns:
            dict: Dictionary containing sentiment label and probabilities
        """
        # Tokenize input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
        # Convert prediction to sentiment label
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_labels[prediction]
        
        # Get probability scores
        probs = probabilities[0].cpu().numpy()
        
        return {
            'sentiment': sentiment,
            'probabilities': {
                'Negative': float(probs[0]),
                'Neutral': float(probs[1]),
                'Positive': float(probs[2])
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for input text')
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pt')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--model_path', type=str, default=model_path,
                      help='Path to the trained model')
    args = parser.parse_args()
    
    if not args.text:
        text = input("Please enter the text to analyze: ")
    else:
        text = args.text
    
    try:
        predictor = SentimentPredictor(model_path=args.model_path)
        result = predictor.predict(text)
        
        print("\nSentiment Analysis Results:")
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print("\nProbabilities:")
        for sentiment, prob in result['probabilities'].items():
            print(f"{sentiment}: {prob:.4f}")
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
