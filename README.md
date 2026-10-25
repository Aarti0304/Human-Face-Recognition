
# Human Face Recognition System

## Project Overview
The Human Face Recognition System is a Python-based application that detects faces and recognizes registered people using OpenCV and the Local Binary Patterns Histograms (LBPH) algorithm. It provides a user-friendly interface built with Streamlit.

## Features
- Face detection using Haar Cascade.
- Face recognition using the LBPH algorithm.
- Register and recognize known faces through a trained model.
- Display "Unknown" for faces that do not match the registered people.
- Upload an image or capture a photo through the application.
- Display recognition results in a modern user interface.

## Technologies Used
- Python
- OpenCV
- NumPy
- Streamlit
- Pillow
- LBPH Face Recognizer
- Haar Cascade Classifier

## Project Structure
```text
Human-Face-Recognition/
├── data/
├── models/
│   ├── haarcascade_frontalface_default.xml
│   ├── face_model.yml
│   └── labels.json
├── screenshots/
├── face_detection.py
├── train_faces.py
├── recognize_faces.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Install Python.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Train the model using your own consented face images:

```bash
python train_faces.py
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## How It Works
1. The application receives an image.
2. The Haar Cascade classifier detects faces.
3. The detected face is processed by the LBPH recognizer.
4. The model predicts the closest registered identity.
5. A threshold determines whether the face is recognized or marked as Unknown.

## Privacy Note
Use face images only with the permission of the people shown. Do not upload personal face datasets or sensitive biometric information to a public repository.

## Project Purpose
This project was developed for educational purposes as an engineering mini-project.
