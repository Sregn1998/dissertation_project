import cv2
import numpy as np
from keras.models import load_model

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

def predict_emotion(video_path, model_path):
    model = load_model(model_path)
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)  # Расширяем размеры массива для соответствия входу модели
    predictions = model.predict(frames)[0]
    return np.argmax(predictions), np.max(predictions)

video_path = 'C:/Users/Sergey/Downloads/Telegram Desktop/2_5465528042312973211 (2).MP4'
model_path = 'C:/Users/Sergey/Desktop/Dis_project/neural_network/models/emotion_video_model.h5'
predicted_class, confidence = predict_emotion(video_path, model_path)
print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
