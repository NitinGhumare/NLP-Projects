# src/data_preprocessing.py
import re
import numpy as np
import gensim
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, vocab_size=20000, max_len=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        words = [w for w in text.split() if w not in stopwords.words("english")]
        return " ".join(words)

    def fit_transform(self, texts):
        texts = [self.clean_text(t) for t in texts]
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        return padded

    def transform(self, texts):
        texts = [self.clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        return padded

def build_embedding_matrix_from_glove(glove_path, word_index, embedding_dim=100):
    """
    glove_path: path to local glove.6B.100d.txt or similar
    word_index: tokenizer.word_index
    Returns embedding_matrix shaped (vocab_size, embedding_dim)
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    vocab_size = min(len(word_index) + 1, 20000)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_embedding_matrix_from_word2vec(w2v_path, word_index, embedding_dim=100):
    """
    Load gensim Word2Vec/KeyedVectors (binary/text) and map to tokenizer index
    """
    kv = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
    vocab_size = min(len(word_index) + 1, 20000)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word in kv:
            embedding_matrix[i] = kv[word]
    return embedding_matrix
