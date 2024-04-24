import numpy as np
import pandas as pd
import librosa
import os
import glob
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Bidirectional

class EmotionVoiceModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.sample_rate = 22050  
        self.num_mfcc = 40  

    def extract_features(self, file_path):
        audio, sample_rate = librosa.load(file_path, sr=self.sample_rate, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.num_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed

    def load_data(self):
        features = []
        labels = []
        all_files = glob.glob(os.path.join(self.data_path, '**/*.wav'), recursive=True)
        print("Files found:", len(all_files))
        for file in all_files:
            file_name = os.path.basename(file)
            emotion = int(file_name.split("-")[2]) - 1
            feature = self.extract_features(file)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
        if features:
            print("Features extracted:", len(features))
        else:
            print("No features extracted.")
        return np.array(features), np.array(labels)

    def build_model(self, num_features):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(1, num_features)),
            Dropout(0.4),
            Bidirectional(LSTM(128)),
            Dropout(0.7),
            Dense(8, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, features, labels):
        features = np.expand_dims(features, 1)  # Расширяем размерность данных для LSTM
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = self.build_model(features.shape[2])
        history = model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_test, y_test))
        return model, history

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Потери на обучении')
        plt.plot(history.history['val_loss'], label='Потери на валидации')
        plt.title('Динамика потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Точность на обучении')
        plt.plot(history.history['val_accuracy'], label='Точность на валидации')
        plt.title('Динамика точности')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, model, save_path='emotion_voice_model.h5'):
        model.save(save_path)

# Создаем экземпляр класса
data_path = "C:\\Users\\Sergey\\Desktop\\Dis_project\\neural_network\\Audio_speach"
emotion_model = EmotionVoiceModel(data_path)

# Загружаем данные
features, labels = emotion_model.load_data()

# Обучаем модель
model, history = emotion_model.train_model(features, labels)

# Визуализируем историю обучения
emotion_model.plot_history(history)

# Сохраняем модель
emotion_model.save_model(model)
