
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_model.yml"
LABELS_PATH = BASE_DIR / "models" / "labels.json"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"

THRESHOLD = 60

st.set_page_config(
    page_title="FaceVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- MODERN DARK UI --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #080d1b 0%, #10172b 55%, #11152b 100%);
        color: #eef2ff;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 27, 0.85);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: #eef2ff;
    }

    .hero {
        padding: 26px 28px;
        border: 1px solid rgba(139, 127, 255, 0.28);
        border-radius: 22px;
        background: linear-gradient(
            120deg, rgba(93, 75, 220, 0.20),
            rgba(37, 99, 235, 0.10)
        );
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #a99fff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 8px 0;
        color: #ffffff;
    }

    .hero-description {
        color: #aab6d3;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(23, 32, 56, 0.88);
        border: 1px solid #293452;
        border-radius: 16px;
        padding: 19px;
        min-height: 120px;
    }

    .metric-label {
        color: #aab6d3;
        font-size: 0.83rem;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 750;
    }

    .status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #34d399;
        margin-right: 8px;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.55);
    }

    .section-heading {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }

    .muted {
        color: #aab6d3;
        font-size: 0.9rem;
    }

    .result-card {
        padding: 15px 17px;
        border-radius: 14px;
        background: rgba(23, 32, 56, 0.88);
        border: 1px solid #293452;
        margin: 9px 0;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #7657f6, #4778f5);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 44px;
        font-weight: 650;
        width: 100%;
    }

    div.stButton > button:hover {
        border: 1px solid #a99fff;
        color: white;
    }

    [data-testid="stFileUploader"],
    [data-testid="stCameraInput"] {
        background: rgba(23, 32, 56, 0.6);
        border: 1px solid #293452;
        border-radius: 14px;
        padding: 12px;
    }

    hr {
        border-color: #293452;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(str(MODEL_PATH))

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if detector.empty():
        raise RuntimeError(
            f"Could not load Haar Cascade: {CASCADE_PATH}"
        )

    return model, names, detector


def get_person_name(label_id, names):
    """Support labels stored as ID-to-name or name-to-ID."""
    name = names.get(str(label_id), names.get(label_id))

    if name is not None:
        return str(name)

    for person_name, person_id in names.items():
        if str(person_id) == str(label_id):
            return str(person_name)

    return f"Person_{label_id}"


def recognize_image(image):
    """Detect faces and label them using the trained LBPH model."""
    rgb_image = np.array(image.convert("RGB"))
    output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(120, 120),
    )

    results = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label_id, distance = recognizer.predict(face_roi)

        person_name = get_person_name(label_id, labels)

        if distance <= THRESHOLD:
            display_name = person_name
            box_color = (52, 211, 153)  # Green
            status = "Recognized"
        else:
            display_name = "Unknown"
            box_color = (80, 120, 255)  # Orange-red in BGR
            status = "Unknown"

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            box_color,
            3,
        )

        cv2.putText(
            output_image,
            display_name,
            (x, max(y - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "name": display_name,
                "distance": float(distance),
                "status": status,
            }
        )

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_rgb, results


# -------------------- APP HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Computer Vision • Deep Learning</div>
        <div class="hero-title">FaceVision AI</div>
        <p class="hero-description">
            Human Face Recognition System · Detect, recognize and verify faces.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL STATUS --------------------

try:
    recognizer, labels, detector = load_model()

    if not isinstance(labels, dict):
        raise ValueError("labels.json must contain a JSON object.")

    model_ready = True
    model_message = "Model loaded and ready"
    registered_count = len(set(labels.values())) if labels else 0

    # Handle labels stored as person-name-to-ID.
    if labels and all(
        isinstance(value, (int, float)) for value in labels.values()
    ):
        registered_count = len(labels)

except Exception as error:
    model_ready = False
    model_message = "Model could not be loaded"
    model_error = str(error)
    registered_count = 0
    recognizer = labels = detector = None

if model_ready:
    status_html = (
        '<span class="status-dot"></span>'
        '<span style="color:#6ee7b7;font-weight:700;">SYSTEM ONLINE</span>'
    )
else:
    status_html = (
        '<span style="color:#fca5a5;font-weight:700;">'
        '● SYSTEM ERROR</span>'
    )

st.markdown(status_html, unsafe_allow_html=True)
st.markdown("")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Registered people</div>
            <div class="metric-value">{registered_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Recognition model</div>
            <div class="metric-value" style="font-size:1.2rem;">
                {"LBPH Ready" if model_ready else "Unavailable"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Face detector</div>
            <div class="metric-value" style="font-size:1.2rem;">
                {"Haar Cascade Ready" if model_ready else "Unavailable"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not model_ready:
    st.error(
        f"{model_message}. Check your model, labels, and cascade files. "
        f"Details: {model_error}"
    )
    st.stop()

st.markdown("")
st.markdown(
    '<div class="section-heading">Recognition workspace</div>'
    '<div class="muted">Upload a photo or capture one using your camera.</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# -------------------- INPUT OPTIONS --------------------

upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📸 Take a Photo"])

uploaded_image = None
input_source = None

with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear photo with a visible face.",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            input_source = "Upload Image"
        except Exception as error:
            st.error(f"Could not open this image: {error}")

with camera_tab:
    camera_file = st.camera_input(
        "Capture a photo",
        help="Keep your face clearly visible and well lit.",
    )

    if camera_file is not None:
        try:
            uploaded_image = Image.open(camera_file).convert("RGB")
            input_source = "Camera"
        except Exception as error:
            st.error(f"Could not process the camera photo: {error}")

# -------------------- RESULTS --------------------

if uploaded_image is not None:
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Recognition results</div>',
        unsafe_allow_html=True,
    )

    try:
        annotated_image, results = recognize_image(uploaded_image)

        preview_col, results_col = st.columns([1.45, 1])

        with preview_col:
            st.markdown("**Face detection preview**")
            st.image(
                annotated_image,
                use_container_width=True,
                caption=f"Input source: {input_source}",
            )

        with results_col:
            st.markdown("**Analysis summary**")

            if not results:
                st.warning(
                    "No face detected. Try a clearer, front-facing photo "
                    "with better lighting."
                )
            else:
                recognized_count = sum(
                    result["status"] == "Recognized" for result in results
                )
                unknown_count = sum(
                    result["status"] == "Unknown" for result in results
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Faces detected</div>
                        <div class="metric-value">{len(results)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Recognized / Unknown</div>
                        <div class="metric-value">
                            <span style="color:#6ee7b7;">{recognized_count}</span>
                            /
                            <span style="color:#fca5a5;">{unknown_count}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for index, result in enumerate(results, start=1):
                    if result["status"] == "Recognized":
                        st.success(
                            f"Face {index}: {result['name']} "
                            f"(distance: {result['distance']:.1f})"
                        )
                    else:
                        st.warning(
                            f"Face {index}: Unknown "
                            f"(distance: {result['distance']:.1f})"
                        )

                st.caption(
                    "LBPH distance is a matching score, not an accuracy "
                    "percentage. Lower scores indicate closer matches."
                )

    except Exception as error:
        st.error(f"Could not process the image: {error}")

else:
    st.info(
        "Your recognition workspace is ready. Upload an image or use "
        "the camera to begin."
    )

# -------------------- FOOTER --------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#8c9ab9;font-size:0.82rem;">
        FaceVision AI · Human Face Recognition Mini Project
        <br>Powered by OpenCV, LBPH and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
