
import cv2
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "face_model.yml"
CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
LABELS_PATH = MODEL_DIR / "labels.json"

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

# Load person names
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

# Load face detector
detector = cv2.CascadeClassifier(str(CASCADE_PATH))

# Lower LBPH confidence distance generally means a closer match.
# This is an initial threshold; we will test and adjust it.
THRESHOLD = 80


def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=7,
    minSize=(100, 100)
)

    if len(faces) == 0:
        print("No face detected.")

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label_id, distance = recognizer.predict(face)

        if str(label_id) in labels and distance < THRESHOLD:
            name = labels[str(label_id)]
            color = (0, 200, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        text = f"{name} ({distance:.1f})"

        cv2.rectangle(
            image, (x, y), (x + w, y + h), color, 2
        )
        cv2.putText(
            image, text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        print(f"Result: {text}")

    return image


def test_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not open the image. Check its path.")
        return

    result = recognize_faces(image)

    cv2.imshow("Face Recognition", result)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read from webcam.")
            break

        result = recognize_faces(frame)
        cv2.imshow("Face Recognition - Webcam", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to an image to recognize")
parser.add_argument(
    "--webcam", action="store_true",
    help="Recognize faces using the webcam"
)
args = parser.parse_args()

if args.webcam:
    test_webcam()
elif args.image:
    test_image(Path(args.image))
else:
    print("Use --image <image_path> or --webcam")
