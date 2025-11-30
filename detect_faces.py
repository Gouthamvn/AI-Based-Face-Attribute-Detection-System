import cv2
from src.utils import load_haar_cascade

class FaceDetector:
    def __init__(self, model_path):
        self.face_cascade = load_haar_cascade(model_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
