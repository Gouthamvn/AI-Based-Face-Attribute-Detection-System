import numpy as np
from tensorflow.keras.models import load_model
import cv2


class EmotionDetector:
    def __init__(self, model_path):
        # Load the model without compiling
        self.model = load_model(model_path, compile=False)
        # Define labels for emotions
        self.labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def predict_emotion(self, face):
        # Resize to 64x64 and convert to grayscale
        face = cv2.resize(face, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Normalize and expand dimensions to match model input
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict emotion
        emotion_idx = np.argmax(self.model.predict(face))
        return self.labels[emotion_idx]
