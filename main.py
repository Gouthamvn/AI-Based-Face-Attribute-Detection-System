from src.real_time_detector import RealTimeDetector

# Paths to model files (updated face model path to 'models/')
FACE_MODEL_PATH = 'models/haarcascade_frontalface_default.xml'
AGE_PROTO_PATH = 'data/age_deploy.prototxt'
AGE_MODEL_PATH = 'data/age_net.caffemodel'
GENDER_PROTO_PATH = 'data/gender_deploy.prototxt'
GENDER_MODEL_PATH = 'data/gender_net.caffemodel'
EMOTION_MODEL_PATH = 'data/emotion_model.hdf5'

def main():
    detector = RealTimeDetector(
        FACE_MODEL_PATH,
        AGE_PROTO_PATH,
        AGE_MODEL_PATH,
        GENDER_PROTO_PATH,
        GENDER_MODEL_PATH,
        EMOTION_MODEL_PATH
    )
    detector.run()

if __name__ == "__main__":
    main()
