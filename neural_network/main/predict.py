import re
import numpy as np
import librosa
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
from AudioToText import AudioToText
from VideoToAudio import VideoToAudio

class DepressionDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = load_model(model_path)
        self.threshold = threshold
        self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    def read_text_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def predict_depression(self, text):
        text = self.preprocess_text(text)
        words = text.split()
        self.tokenizer.fit_on_texts(words)
        sequences = self.tokenizer.texts_to_sequences(words)
        padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')
        predictions = self.model.predict(padded_sequences)
        average_prediction = np.mean(predictions)
        return average_prediction

    def assess_mental_health(self, average_prediction):
        if average_prediction < self.threshold:
            return "Отклонений не обнаружено."
        else:
            return "У вас наблюдаются расстройства психического характера. Рекомендую вам обратиться к специалисту за помощью."

class EmotionVoicePredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.sample_rate = 22050
        self.num_mfcc = 40

    def extract_features(self, file_path):
        print(file_path)
        print("123")
        audio, _ = librosa.load(file_path, sr=self.sample_rate, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.num_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return np.expand_dims(np.expand_dims(mfccs_processed, axis=0), axis=0)

    def predict_emotion(self, file_path):
        features = self.extract_features(file_path)
        predictions = self.model.predict(features)[0]
        print("Raw predictions:", predictions)
        emotions = ["нейтральные", "спокойные", "счастливые", "грустные", "сердитые", "испуганные", "отвращение", "удивленные"]
        percentages = {emotions[i]: p * 100 for i, p in enumerate(predictions)}
        return percentages

    def visualize_emotions(self, percentages):
        emotions = list(percentages.keys())
        values = list(percentages.values())
        plt.figure(figsize=(10, 5))
        plt.bar(emotions, values, color='blue')
        plt.xlabel('Emotions')
        plt.ylabel('Percentage')
        plt.title('Emotional Distribution in Audio')
        plt.xticks(rotation=45)
        plt.savefig(image_file_path)  
        plt.close()  
        

class PredictionService:  
    def generate_html(self, text_analysis, emotions_percentages):
        image_path = "/static/emotion_chart.png"
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Results</title>
        </head>
        <body>
            <h1>Analysis Results</h1>
            <h2>Text Analysis</h2>
            <p>{text_analysis}</p>
            <h2>Emotional Distribution in Audio</h2>
            <img src="{image_path}" alt="Emotional Distribution">
            <ul>
        """
        for emotion, percentage in emotions_percentages.items():
            html_content += f"<li>{emotion}: {percentage:.2f}%</li>"

        html_content += """
            </ul>
        </body>
        </html>
        """
        return html_content
    
    def start_prediction(self, video):
        audio_exemplar = VideoToAudio(video.filename, audio_file_path) 
        audio_path = audio_exemplar.extract_audio()
        audio_percentages = audio_predictor.predict_emotion(audio_file_path)
        audio_predictor.visualize_emotions(audio_percentages)

        text_exemplar = AudioToText(audio_file_path) 
        text_path = text_exemplar.recognize_audio()
        text = text_detector.read_text_from_file(text_file_path)
        text_prediction = text_detector.predict_depression(text)
        text_result = text_detector.assess_mental_health(text_prediction)

        html_content = self.generate_html(self, text_analysis = text_result, emotions_percentages = audio_percentages)
        return html_content


text_file_path = "./sample_video/result.txt"
audio_file_path = "./1.wav"
image_file_path = "./neural_network/main/images/emotion_chart.png"

text_detector = DepressionDetector(model_path="./neural_network/models/depression_model.h5")
audio_predictor = EmotionVoicePredictor(model_path="./neural_network/models/emotion_voice_model.h5")

# text = text_detector.read_text_from_file(text_path)
# text_prediction = text_detector.predict_depression(text)
# text_result = text_detector.assess_mental_health(text_prediction)

# audio_percentages = audio_predictor.predict_emotion(audio_file_path)
# audio_predictor.visualize_emotions(audio_percentages)

# print("Текстовый анализ:", text_result)
# print("Процентное содержание эмоций в аудио:", audio_percentages)
