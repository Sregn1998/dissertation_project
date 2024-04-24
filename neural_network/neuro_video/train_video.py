import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras.utils import to_categorical
import cProfile
import pstats

def parse_label_from_filename(filename):
    parts = filename.split('-')
    emotion_code = int(parts[2])  # Эмоции находятся на третьем месте в имени файла
    return emotion_code - 1  # Переводим код в индекс (0-индексирование)

def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(frame_count / num_frames), 1)
    for _ in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, _ * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame.astype('float32') / 255.0)
    cap.release()
    return np.array(frames)

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        TimeDistributed(GlobalAveragePooling2D(), input_shape=(20, 224, 224, 3)),
        LSTM(128),
        Dropout(0.5),
        Dense(8, activation='softmax')  # предполагается 8 классов эмоций
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    video_directory = 'B:/Emotions_video/speach'
    video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
    labels = [parse_label_from_filename(os.path.basename(f)) for f in video_files]
    model = create_model()
    for video_path, label in zip(video_files, labels):
        frames = extract_frames(video_path)
        label = to_categorical(label, num_classes=8)
        frames = np.expand_dims(frames, axis=0)
        label = np.expand_dims(label, axis=0)
        model.train_on_batch(frames, label)
    model.save('emotion_video_model.h5')

with cProfile.Profile() as pr:
    main()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME).print_stats(10)