# AI-Based-Face-Attribute-Detection-System
The AI-Based Face Attribute Detection System is a real-time computer vision project that detects Age, Gender, and Emotion from live video input using deep learning models. The system uses OpenCV, TensorFlow/Keras, and pre-trained models to perform fast and accurate predictions on detected faces.
Features

👤 Real-time face detection using Haar Cascade

🎭 Emotion detection (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)

🧑‍🤝‍🧑 Gender prediction (Male / Female)

🎂 Age group prediction (0–100+ age range)

📹 Works with webcam or video files

⚡ Fast inference using lightweight deep learning models

🧠 Uses pre-trained deep learning models for accuracy

🖥 Clean UI overlays with bounding boxes & labels

## Tech Stack

Python

OpenCV

TensorFlow / Keras

NumPy

Pre-trained CNN Models

Haarcascade Frontal Face Detection
## Project Stucture
AI-Based-Face-Attribute-Detection/
│── models/
│   ├── age_model.h5
│   ├── emotion_model.h5
│   ├── gender_model.h5
│── haarcascade/
│   └── haarcascade_frontalface_default.xml
│── src/
│   ├── detect.py
│── README.md
│── requirements.txt

## Dataset Information

This project uses publicly available datasets such as:

UTKFace Dataset – Age & Gender
FER2013 Dataset – Emotion Recognition
## Installation
1. Clone the repository
git clone https://github.com/yourusername/AI-Based-Face-Attribute-Detection-System.git
cd AI-Based-Face-Attribute-Detection-System

2. Install dependencies
pip install -r requirements.txt

3. Run the project
python src/detect.py

## How It Works

Your webcam feed is captured

Faces are detected using Haar Cascade

Each face is cropped and passed through 3 models:

Age prediction

Gender prediction

Emotion prediction

The final result is displayed on the screen with bounding boxes
