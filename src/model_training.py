import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from src.data_preprocessing import prepare_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        return {
            'input_ids': 
            'attention_mask': 
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions)
    return avg_loss, accuracy, report

def main():
    # Hyperparameters
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {DEVICE}")

    df = pd.read_csv('src/reviews.csv')

    texts = df['content'].values  
    # Convert scores to binary sentiment (1-2 → 0 (negative), 4-5 → 1 (positive))
    df['sentiment'] = df['score'].apply(lambda x: 0 if x <= 2 else 1 if x >= 4 else None)
    # Remove (score = 3) as they are neutral 
    df = df.dropna(subset=['sentiment'])
    
    texts = df['content'].values
    labels = df['sentiment'].values.astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,  # binary class 
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, DEVICE)
        logger.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Evaluate
        val_loss, val_accuracy, val_report = evaluate(model, val_loader, DEVICE)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        logger.info("\nClassification Report:\n" + val_report)

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pt')
            logger.info("Saved new best model")

    logger.info("Training completed!")

if __name__ == "__main__":
    main()
