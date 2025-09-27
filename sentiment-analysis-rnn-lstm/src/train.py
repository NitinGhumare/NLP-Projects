# src/train.py (key parts)
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import TextPreprocessor, build_embedding_matrix_from_glove
from model import build_lstm_model
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "sentiment_data_large.csv")  # use big file
ARTIFACT_MODEL = os.path.join("artifacts", "sentiment_lstm.keras")
TOKENIZER_PATH = os.path.join("artifacts", "tokenizer.pkl")
GLOVE_PATH = os.path.join("artifacts", "glove.6B.100d.txt")  # place if available

# params
VOCAB_SIZE = 20000
MAX_LEN = 100
EMBED_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64

df = pd.read_csv(DATA_PATH)
X = df["text"].values
y = df["label"].values

processor = TextPreprocessor(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
X_padded = processor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Optionally build embedding matrix if GloVe available
embedding_matrix = None
if os.path.exists(GLOVE_PATH):
    print("Loading GloVe embeddings (this may take a while)...")
    embedding_matrix = build_embedding_matrix_from_glove(GLOVE_PATH, processor.tokenizer.word_index, embedding_dim=EMBED_DIM)
    print("GloVe loaded.")

model = build_lstm_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, max_len=MAX_LEN,
                         embedding_matrix=embedding_matrix, trainable_embeddings=False)
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# Save model and tokenizer
model.save(ARTIFACT_MODEL)
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(processor.tokenizer, f)

# Plot training curves
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join("artifacts", "loss_curve.png"))
plt.close()

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join("artifacts", "acc_curve.png"))
plt.close()

print("Training complete. Artifacts saved in artifacts/")
