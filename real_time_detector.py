import cv2
from collections import deque, Counter
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from src.detect_faces import FaceDetector
from src.age_gender_detection import AgeGenderDetector
from src.emotion_detection import EmotionDetector
from src.utils import draw_label

class RealTimeDetector:
    def __init__(self, face_model_path, age_proto_path, age_model_path, gender_proto_path, gender_model_path, emotion_model_path):
        self.face_detector = FaceDetector(face_model_path)
        self.age_gender_detector = AgeGenderDetector(age_proto_path, age_model_path, gender_proto_path, gender_model_path)
        self.emotion_detector = EmotionDetector(emotion_model_path)

        # Buffers to store recent predictions for smoothing
        self.gender_buffer = deque(maxlen=10)
        self.age_buffer = deque(maxlen=10)
        self.emotion_buffer = deque(maxlen=10)

        # Store last stable prediction
        self.last_gender = "Unknown"
        self.last_age = "Unknown"
        self.last_emotion = "Unknown"

        # Emotion count dictionary for graph
        self.emotion_count = {}

        # Create folder for screenshots
        os.makedirs("screenshots", exist_ok=True)

    def get_stable_prediction(self, buffer, last_value):
        if buffer:
            return Counter(buffer).most_common(1)[0][0]
        return last_value

    def save_screenshot(self, frame, emotion):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshots/{emotion}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved: {filename}")

    def update_emotion_count(self, emotion):
        self.emotion_count[emotion] = self.emotion_count.get(emotion, 0) + 1

    def plot_emotion_graph(self):
        plt.clf()
        emotions = list(self.emotion_count.keys())
        counts = list(self.emotion_count.values())
        plt.bar(emotions, counts, color='skyblue')
        plt.xlabel("Emotions")
        plt.ylabel("Count")
        plt.title("Real-Time Emotion Count")
        plt.pause(0.001)  # Allows real-time update

    def run(self):
        cap = cv2.VideoCapture(0)
        plt.ion()  # Enable interactive mode for live graph

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.face_detector.detect_faces(frame)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]

                    # Predict values
                    gender, age = self.age_gender_detector.predict_age_gender(face)
                    emotion = self.emotion_detector.predict_emotion(face)

                    # Add predictions to buffers
                    self.gender_buffer.append(gender)
                    self.age_buffer.append(age)
                    self.emotion_buffer.append(emotion)

                    # Get stable values
                    self.last_gender = self.get_stable_prediction(self.gender_buffer, self.last_gender)
                    self.last_age = self.get_stable_prediction(self.age_buffer, self.last_age)
                    self.last_emotion = self.get_stable_prediction(self.emotion_buffer, self.last_emotion)

                    # Update emotion count & graph
                    self.update_emotion_count(self.last_emotion)
                    self.plot_emotion_graph()

                    # Save screenshot for specific emotions
                    if self.last_emotion in ["Happy", "Sad"]:
                        self.save_screenshot(frame, self.last_emotion)

                    # Draw face box & label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    draw_label(frame, f"{self.last_gender}, Age: {self.last_age}, Emotion: {self.last_emotion}", x, y - 10)

            # ==== DATE & TIME OVERLAY ====
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_time, (frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Face, Age, Gender, and Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
