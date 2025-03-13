import streamlit as st
import torch
from inference import SentimentPredictor  # Importing the predictor class
import json

st.set_page_config(page_title="Sentiment Analysis App", page_icon="🔍")

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return SentimentPredictor()

predictor = load_predictor()


def save_prediction(sentence, sentiment):
    dict_content = {}
    with open("predictions.json", "r") as file:
        dict_content = json.load(dict_content)

    prediction_id = len(dict_content) + 1

    with open("predictions.json", "w") as file:
        json.dump({prediction_id : [sentence, sentiment]}, file, indent=4)

    return [prediction_id , [sentence, sentiment]]


def main():
    st.title("🔍 Sentiment Analysis App")
    st.markdown("Welcome to the Sentiment Analysis App! Enter a sentence and discover its sentiment. 😊")

    st.markdown("### 📝 How does it work?")
    st.markdown("This app uses a BERT-based model fine-tuned for sentiment analysis. The model classifies the input sentence as **Positive**, **Neutral**, or **Negative**. 🚀")

    txt_input = st.text_area("✍️ Enter your text here:", height=150)

    # Global storage for predictions
    predictions = {}

    if st.button("🔎 Analyze Sentiment"):
        if txt_input.strip():
            try:
                result = predictor.predict(txt_input)
                sentiment = result['sentiment']
                probabilities = result['probabilities']

                st.success(f"🎉 Predicted Sentiment: {sentiment}")
                st.write("### 🔢 Probabilities:")
                st.write(f"- Negative: {probabilities['Negative']:.4f}")
                st.write(f"- Neutral: {probabilities['Neutral']:.4f}")
                st.write(f"- Positive: {probabilities['Positive']:.4f}")

                prediction = save_prediction(txt_input, sentiment)
                predictions[prediction[0]] = predictions[prediction[1]]
                
                st.info(f"✅ Prediction saved with ID {len(predictions)}.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("---")
    st.markdown("💡 **Credits:** Created with ❤️ by Your Name. Built using BERT and Streamlit.")

    if st.button("📂 View Saved Predictions"):
        try:
            with open("predictions.json", "r") as file:
                predictions = json.load(file)

            if predictions:
                st.write("### 💾 Saved Predictions:")
                for key, value in predictions.items():
                    st.write(f"ID: {key} | Text: {value[0]} | Sentiment: {value[1]}")
            else:
                st.info("No saved predictions yet.")
        except (FileNotFoundError, json.JSONDecodeError):
            st.info("No saved predictions yet.")

if __name__ == "__main__":
    main()