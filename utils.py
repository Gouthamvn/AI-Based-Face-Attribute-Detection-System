import cv2

def load_haar_cascade(cascade_path):
    return cv2.CascadeClassifier(cascade_path)

def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
