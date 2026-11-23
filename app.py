import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Intelligence | AI Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CLASSY DARK-MODE CSS ---
st.markdown("""
<style>
    /* Dark Obsidian Background */
    .stApp {
        background: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }

    /* Sleek Glowing Title */
    h1 {
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 5px;
    }

    /* Subtitle */
    p {
        color: #9ca3af;
        text-align: center;
        font-size: 1.05rem;
    }

    /* Glassmorphism Upload Box */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(168, 85, 247, 0.8);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
    }

    /* Cards / Metrics / Containers */
    .stSuccess, .stError {
        background: rgba(17, 24, 39, 0.9);
        border-radius: 16px;
        padding: 20px;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Image Preview Frame */
    .stImage img {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 30px rgba(0,0,0,0.6);
    }
</style>
""", unsafe_allow_html=True)

FILE_ID = "14tpgC0bkDan9Z_wB5SxdO7RvhpmBNVuk"
MODEL_PATH = "deepfake_model.pth"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("⚡ Initializing Neural Weights... Please wait."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        )
    except Exception as e:
        st.error(f"Error loading model weights: {e}")

    model.eval()
    return model


def extract_face(pil_image):
    """Detects and crops the largest face with a small margin for accurate deepfake inference."""
    try:
        if not hasattr(cv2, "data"):
            return pil_image

        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if not os.path.exists(cascade_path):
            return pil_image

        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return pil_image

        img_np = np.array(pil_image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return pil_image

        # Pick the largest detected face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Add 20% padding around the face
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_np.shape[1], x + w + pad_w)
        y2 = min(img_np.shape[0], y + h + pad_h)

        return Image.fromarray(img_np[y1:y2, x1:x2])
    except Exception:
        return pil_image


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.markdown("<h1>🛡️ Deepfake Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Advanced PyTorch ResNet18 biometric verification framework for real-time artifact detection.</p>",
    unsafe_allow_html=True,
)
st.write("")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Target Image Analysis", use_container_width=True)

        with st.spinner("🔍 Detecting face & executing inference pass..."):
            face_img = extract_face(image)
            input_tensor = transform(face_img).unsqueeze(0)

            with torch.no_grad():
               with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Index 0 = Fake ('training_fake'), Index 1 = Real ('training_real')
            fake_prob = probs[0].item() * 100
            real_prob = probs[1].item() * 100
            top_conf = max(real_prob, fake_prob)
            result_class = "Real" if real_prob >= fake_prob else "Fake"
            conf_score = confidence.item() * 100

        st.markdown("<br>", unsafe_allow_html=True)
        if result_class == "Real":
            st.success(
                f"✦ **Verification Status: REAL** \n\nConfidence: **{conf_score:.2f}%**"
            )
        else:
            st.error(
                f"✦ **Verification Status: FAKE** \n\nConfidence: **{conf_score:.2f}%**"
            )
