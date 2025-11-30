import cv2
import numpy as np

class AgeGenderDetector:
    def __init__(self, age_proto_path, age_model_path, gender_proto_path, gender_model_path):
        # Load models for age and gender
        self.age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)

        # Define labels
        self.gender_labels = {0: 'Male', 1: 'Female'}
        self.age_ranges = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    def predict_age_gender(self, face):
        # Prepare the blob from face image for the network
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        self.gender_net.setInput(blob)
        gender_idx = np.argmax(self.gender_net.forward())
        gender = self.gender_labels.get(gender_idx, 'Unknown')

        # Predict age
        self.age_net.setInput(blob)
        age_idx = np.argmax(self.age_net.forward())
        age = self.age_ranges[age_idx]

        return gender, age
