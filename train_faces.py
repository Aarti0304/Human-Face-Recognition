import cv2
import json
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "known_faces"
MODEL_DIR = BASE_DIR / "models"

CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = MODEL_DIR / "face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"

# Check required files
if not CASCADE_PATH.exists():
    raise FileNotFoundError(
        f"Face detection file not found: {CASCADE_PATH}"
    )

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Known faces folder not found: {DATA_DIR}"
    )

# Load face detector
face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError("Could not load the face detection model.")

# Check LBPH availability
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "LBPH is unavailable. Check that opencv-contrib-python "
        "is installed in your project virtual environment."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()

training_faces = []
training_labels = []
labels = {}
label_id = 0

# Read each person's folder
for person_folder in sorted(DATA_DIR.iterdir()):
    if not person_folder.is_dir():
        continue

    labels[label_id] = person_folder.name
    image_files = sorted(
    file for file in person_folder.iterdir()
    if file.is_file()
    and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

    if not image_files:
        print(f"No images found in {person_folder.name}")
        label_id += 1
        continue

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(detected_faces) == 0:
            print(f"No face detected in: {image_path.name}")
            continue

        # Use the largest detected face
        x, y, w, h = max(
            detected_faces, key=lambda box: box[2] * box[3]
        )

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        training_faces.append(face)
        training_labels.append(label_id)

        print(f"Added: {person_folder.name}/{image_path.name}")

    label_id += 1

# Train and save the model
if not training_faces:
    raise RuntimeError(
        "No faces were collected. Check your photos and try again."
    )

recognizer.train(
    training_faces,
    __import__("numpy").array(training_labels, dtype="int32")
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
recognizer.save(str(MODEL_PATH))

with open(LABELS_PATH, "w", encoding="utf-8") as file:
    json.dump(labels, file, indent=4)

print("\nTraining completed successfully!")
print(f"Photos used: {len(training_faces)}")
print(f"People registered: {len(set(training_labels))}")
print(f"Model saved to: {MODEL_PATH}")
print(f"Labels saved to: {LABELS_PATH}")
