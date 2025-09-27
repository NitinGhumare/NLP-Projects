# Sentiment Analysis using RNN/LSTM

This project implements a deep learning model for text sentiment classification using LSTM layers.

## Features
- Preprocessing: tokenization, stopword removal, padding
- Word embeddings with Keras Tokenizer
- LSTM model for binary classification
- Dropout layers to reduce overfitting
- Saved model + tokenizer for inference

## Tech Stack
Python, TensorFlow, Keras, NLTK, NumPy, Matplotlib

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Run inference
python src/predict.py
