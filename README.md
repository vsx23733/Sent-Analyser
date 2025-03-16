# Sentiment Analysis App 🌟

Welcome to the Sentiment Analysis App! This project leverages a BERT-based model to predict the sentiment of a given sentence as Positive, Neutral, or Negative. The application is built using Streamlit, providing an intuitive and interactive user experience.

## 🚀 Features

  - Real-Time Sentiment Analysis: Enter any text and get instant sentiment predictions.
  - BERT-Based Model: Uses a fine-tuned BERT model for accurate sentiment classification.
  - Interactive UI: Built with Streamlit for a smooth and user-friendly experience.
  - Prediction History: Saves predictions with an ID and the corresponding sentiment.
  - Exportable Predictions: Predictions are saved in a JSON file for easy access and analysis.

## 🔥 Demo

Check out the demo of the app:

[![Demo of the Sentiment-Analysis App](https://img.youtube.com/vi/UTA-s4sFP_Y/0.jpg)](https://youtu.be/UTA-s4sFP_Y)


## 📝 Installation

1. Clone the repository:
   ```
   git clone https://github.com/asseromar/Sent-Analysis.git
   cd Sent-Analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## 🧠 How It Works

1. The user inputs a sentence through the UI.
2. The sentence is processed by the BERT-based model to predict the sentiment.
3. The prediction and probabilities are displayed on the screen.
4. The result is saved in predictions.json for future reference.


## 📂 Saved Predictions

The predictions are stored in predictions.json in the following format:
```
{
    "1": ["I love this app!", "Positive"],
    "2": ["It is okay, not the best.", "Neutral"],
    "3": ["I hate waiting!", "Negative"]
}
```

## 🧪 Running Tests

Unit tests are located in the tests/unit directory. You can run them using:
```
pytest tests/unit
```

## 📝 Technical Details and Results

This section covers the technical details of the model, training process, evaluation metrics, and detailed performance analysis.

### Data Extraction and Processing

The data extraction and processing workflow is essential for preparing the dataset for sentiment analysis using the BERT-based model. This part of the project is responsible for loading the raw data, processing it into a usable format, and transforming the reviews into sentiment labels. Below is a detailed breakdown of the functions and classes involved in the data extraction and processing process.
- Data Extraction
    The first step is to load the data from the provided CSV file containing the Google reviews. The load_data() function reads the CSV file into a pandas DataFrame.
  
- Sentiment Labeling
    To convert the ratings from the reviews into sentiment scores, the to_sentiment() function maps the ratings to a corresponding sentiment:
      Negative sentiment (0) for ratings 1-2.
      Neutral sentiment (1) for rating 3.
      Positive sentiment (2) for ratings 4-5.
    The create_sentiment_column() function applies the to_sentiment() function to the entire dataset and adds a new column for sentiment labels.
  
- Data Processing for Model Input
    Once the sentiment labels are added to the DataFrame, the next step is to preprocess the reviews so they can be fed into the BERT model. The function preprocess_reviews_processing() tokenizes the review content and converts the data      into the format expected by BERT. It returns:
         - input_ids: The tokenized input data for BERT.
         - attention_mask: The attention mask to tell BERT where to focus during attention mechanisms.
         - labels: The sentiment labels for each review.

- Dataset and Dataloader
    To efficiently load the data into the model, we use the SentimentDataset class, which wraps the preprocessed data into a PyTorch dataset. The function prepare_dataloader() manages the complete pipeline for loading, processing, and        splitting the dataset into training and testing sets, then returning PyTorch DataLoaders for each and the function returns the train_loader and the test_loader.

    


### Model Architecture

The model architecture for this sentiment analysis task is based on BERT (Bidirectional Encoder Representations from Transformers), which is pre-trained on vast amounts of text data and fine-tuned for downstream tasks such as sentiment classification. In this project, we use BERT for Sequence Classification from the Hugging Face Transformers library to build a robust model for classifying the sentiment of Google reviews.

- Pretrained BERT Model (BertForSequenceClassification)
    The core of the model is BERT, specifically the BertForSequenceClassification variant, which is designed to perform sequence classification tasks, such as sentiment analysis. This model is pretrained on the English Wikipedia and          BookCorpus datasets, allowing it to understand general language patterns, semantics, and syntax.
    The model uses the [CLS] token at the beginning of the input sequence to aggregate information and predict the class for the entire sequence (sentiment). The output is passed through a classification head (a dense layer) to generate      the final sentiment prediction.

- Classification Head
    The classification head is a dense layer that takes the output from the [CLS] token (which represents the entire review) and performs a final classification into one of three sentiment categories: Negative (0), Neutral (1), and           Positive (2). The BertForSequenceClassification model has a linear classification layer on top of the BERT transformer, which is the final layer used to predict the sentiment label. The output layer has 3 units corresponding to the 3     sentiment classes.

- Optimizer and Loss Function
    Optimizer: We use the AdamW optimizer (a variant of Adam optimized for weight decay), which is widely used for training transformer models. The learning rate is set to 2e-5, a typical value for fine-tuning BERT.
    Loss Function: The model uses cross-entropy loss, which is appropriate for multi-class classification tasks. The loss is computed based on the predicted class probabilities and the true sentiment labels.

- Training and Evaluation Loop
    Training: In each training epoch, the model goes through the training dataset (using the train_epoch function). For each batch, the model computes the loss, performs backpropagation, and updates the weights.
    Evaluation: After every epoch, the model is evaluated on the validation set using the evaluate function. The evaluation calculates the loss and accuracy, and generates a classification report that provides detailed metrics such as        precision, recall, and F1-score for each sentiment class.

- Hyperparameters:
    ```
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    ```

- Model Saving
    After each evaluation, the model checks if the current validation accuracy is the best it has achieved so far. If so, the model is saved to the disk as best_model.pt in a directory named models. This ensures that the best-performing      model during training is preserved for later use.

- Device Management (GPU/CPU)
    The model is designed to work on either a GPU (if available) or a CPU. The device is determined dynamically based on the hardware, and the model, as well as the data, is moved to the appropriate device using
    ```
    .to(device).
    ```



### Training and Inference Results

The model was trained on a dataset of Google reviews, and we evaluated its performance based on several metrics, including training loss, validation loss, and classification accuracy.

- Training Loss and Accuracy
    During the training process, we monitored the training loss and accuracy for each epoch. The loss decreased progressively, indicating that the model was learning from the data. The accuracy also improved     as the training proceeded, reflecting the model's increasing ability to correctly classify sentiments. After each epoch, we evaluated the model on the validation set to measure its performance more           comprehensively. We used metrics such as precision, recall, and F1-score for each sentiment class (Negative, Neutral, and Positive).
  
    Epoch 1:
  
    ![Training Logs](assets/images/training_log_1.png)

    Epoch 1:
  
    ![Training Logs](assets/images/training_log_2.png)

    Epoch 1:
  
    ![Training Logs](assets/images/training_log_3.png)


- Inference Logs
    Finally, during the inference phase, the model was used to predict the sentiment of new reviews. The logs below show the model's predictions on sample inputs, including the predicted sentiment and the corresponding probabilities for      each class.

    Positive sentiment prediction inference: 

    ![Inference Logs](assets/images/inference_log_1.png)

    Negative sentiment prediction inference: 

    ![Inference Logs](assets/images/inference_log_2.png)



## 🙏 Credits

- Created with ❤️ by Axel & Asser.
- Built using Python, Streamlit, PyTorch, and Transformers.
- Model based on BERT from the Hugging Face library.

## 🌐 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 💡 Future Improvements

  - Integrate more sentiment labels for fine-grained classification.
  - Add multi-language support.
  - Enhance model accuracy with fine-tuning on additional datasets.
  - Improve the UI with more visualizations and insights.
