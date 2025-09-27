import sys
import pickle
from tensorflow.keras.models import load_model
from data_preprocessing import TextPreprocessor

# Load model & tokenizer
model = load_model("artifacts/sentiment_lstm.h5")
with open("artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Recreate processor
processor = TextPreprocessor(vocab_size=5000, max_len=100)
processor.tokenizer = tokenizer

def predict_sentiment(text):
    padded = processor.transform([text])
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

if __name__ == "__main__":
    # If user gives input from command line
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        print(f"Input: {input_text}")
        print("Prediction:", predict_sentiment(input_text))
    else:
        # Default examples
        samples = [
            "The movie was amazing, I loved it!",
            "Worst film ever. Waste of time.",
            "Quite okay, some good parts and some bad.",
            "Fantastic performance and great story"
        ]
        for s in samples:
            print(f"{s} -> {predict_sentiment(s)}")
