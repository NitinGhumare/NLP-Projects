# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_lstm_model(vocab_size, embedding_dim=100, max_len=100, embedding_matrix=None, trainable_embeddings=False):
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=trainable_embeddings))
    else:
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
