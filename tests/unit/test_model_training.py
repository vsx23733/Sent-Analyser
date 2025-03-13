import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from src.data_processing import prepare_dataloader 
import os
from src.model_training import *

# Test if the model loads and initializes correctly
def test_model_initialization():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    )
    assert model is not None, "Model failed to initialize"
    assert model.num_labels == 3, f"Expected 3 labels, got {model.num_labels}"

# Test if the tokenizer loads correctly
def test_tokenizer_initialization():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    assert tokenizer is not None, "Tokenizer failed to initialize"

# Test the dataloader preparation
def test_dataloader():
    # path_to_data = os.path.join(os.path.dirname(__file__), "../src/reviews.csv")
    train_loader, val_loader = prepare_dataloader("src/reviews.csv")
    assert isinstance(train_loader, DataLoader), "Train loader is not a DataLoader"
    assert isinstance(val_loader, DataLoader), "Validation loader is not a DataLoader"
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"

# Test if the training loop works without errors
def test_train_epoch():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    sample_input = tokenizer("This is a test sentence.", padding=True, truncation=True, return_tensors="pt", max_length=128)
    sample_label = torch.tensor([0]).unsqueeze(0)  # assuming 0 is a valid label
    batch = {'input_ids': sample_input['input_ids'], 'attention_mask': sample_input['attention_mask'], 'labels': sample_label}

    train_loader = DataLoader([batch], batch_size=1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    (train_loss, train_accuracy) = train_epoch(model, train_loader, optimizer, DEVICE)
    
    assert isinstance(train_loss, float), f"Expected loss to be a float, got {type(train_loss)}"
    assert isinstance(train_accuracy, float), f"Expected accuracy to be a float, got {type(train_accuracy)}"
    assert train_loss >= 0, "Loss should not be negative"
    assert 0 <= train_accuracy <= 1, "Accuracy should be between 0 and 1"

# Test if the model is saved correctly after training
def test_model_saving():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sample_input = tokenizer("This is a test sentence.", padding=True, truncation=True, return_tensors="pt", max_length=128)
    sample_label = torch.tensor([0]).unsqueeze(0)  # assuming 0 is a valid label
    batch = {'input_ids': sample_input['input_ids'], 'attention_mask': sample_input['attention_mask'], 'labels': sample_label}

    train_loader = DataLoader([batch], batch_size=1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, DEVICE)

    torch.save(model.state_dict(), 'test_model.pt')
    assert os.path.exists('test_model.pt'), "Model was not saved correctly"

# Test if evaluation produces expected results
def test_evaluate():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sample_input = tokenizer("This is a test sentence.", padding=True, truncation=True, return_tensors="pt", max_length=128)
    sample_label = torch.tensor([0]).unsqueeze(0)  # assuming 0 is a valid label
    batch = {'input_ids': sample_input['input_ids'], 'attention_mask': sample_input['attention_mask'], 'labels': sample_label}

    val_loader = DataLoader([batch], batch_size=1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)

    val_loss, val_accuracy, val_report = evaluate(model, val_loader, DEVICE)
    
    assert isinstance(val_loss, float), f"Expected loss to be a float, got {type(val_loss)}"
    assert isinstance(val_accuracy, float), f"Expected accuracy to be a float, got {type(val_accuracy)}"
    assert isinstance(val_report, str), f"Expected report to be a string, got {type(val_report)}"
    assert val_loss >= 0, "Loss should not be negative"
    assert 0 <= val_accuracy <= 1, "Accuracy should be between 0 and 1"

