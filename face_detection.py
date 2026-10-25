
import cv2
import argparse
from pathlib import Path

# Load OpenCV's pre-trained face detector
from pathlib import Path

cascade_path = str(
    Path(__file__).parent
    / "models"
    / "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load the face detection model.")


def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    
    faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(80, 80)
)


    # Draw a rectangle around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return image, len(faces)


def detect_in_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Could not open the image.")
        print("Check the image path and file name.")
        return

    result, count = detect_faces(image)

    print(f"Faces detected: {count}")

    cv2.imshow("Human Face Recognition - Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_with_webcam():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Check camera permissions or whether another app is using it.")
        return

    print("Webcam started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            result, count = detect_faces(frame)

            cv2.putText(
                result,
                f"Faces: {count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Human Face Recognition - Webcam", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Face Detection System"
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Path to the image file"
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Start live webcam detection"
    )

    args = parser.parse_args()

    if args.image:
        detect_in_image(args.image)
    elif args.webcam:
        detect_with_webcam()
    else:
        parser.error("Use --image IMAGE_PATH or --webcam")
