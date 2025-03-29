import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from src.data_processing import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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

    train_loader, val_loader = prepare_dataloader('reviews.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)



    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

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
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pt')
            logger.info("Saved new best model")

    logger.info("Training completed!")

if __name__ == "__main__":
    main()