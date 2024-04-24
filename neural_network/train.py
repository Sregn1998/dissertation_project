import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import BatchNormalization

class SuicideDetectionModelTrainer:
    def __init__(self, data_path, max_length=100, num_words=10000):
        self.data_path = data_path
        self.max_length = max_length
        self.num_words = num_words

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(data["text"])
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(data["text"])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
        labels = np.array(data["class"].replace({"suicide": 1, "non-suicide": 0}))
        return padded, labels

    def split_data(self, padded, labels, train_size=0.7):
        train_size = int(len(padded) * train_size)
        train_padded = padded[:train_size]
        train_labels = labels[:train_size]
        val_padded = padded[train_size:]
        val_labels = labels[train_size:]
        return train_padded, train_labels, val_padded, val_labels

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_words, 128, input_length=self.max_length),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        return model

    def train_model(self, model, train_padded, train_labels, val_padded, val_labels, epochs=5):
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = model.fit(train_padded, train_labels, epochs=epochs, validation_data=(val_padded, val_labels), verbose=2)
        return history

    def save_model(self, model, save_path):
        model.save(save_path)

data_path = "C:\\Users\\Sergey\\Desktop\\Dis_project\\neural_network\\Suicide_Detection.csv"

trainer = SuicideDetectionModelTrainer(data_path)

data = trainer.load_data()

padded, labels = trainer.preprocess_data(data)

train_padded, train_labels, val_padded, val_labels = trainer.split_data(padded, labels)

model = trainer.build_model()

history = trainer.train_model(model, train_padded, train_labels, val_padded, val_labels)

save_path = "depression_model.h5"
trainer.save_model(model, save_path)