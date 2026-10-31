import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Detector ✨",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- AESTHETIC CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    h1 {
        color: #d87093;
        text-align: center;
        font-weight: 800;
        letter-spacing: -0.5px;
        padding-bottom: 0px;
    }
    
    /* Subtitle / Description */
    p {
        color: #6c757d;
        text-align: center;
        font-size: 1.05rem;
    }

    /* File Uploader Box Styling */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #ffb6c1;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #db7093;
    }

    /* Success / Error Result Boxes */
    .stSuccess, .stError {
        border-radius: 15px;
        padding: 15px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Center images */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

FILE_ID = "1PUxOsAbMjrb8H5zZmRksWTcgzFzOiIfA"
MODEL_PATH = "deepfake_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🌸 Downloading model weights... Just a moment!"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                
    # Initialize ResNet18 architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        
    model.eval()
    return model

model = load_model()

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- APP UI HEADER ---
st.markdown("<h1>🛡️ Deepfake Face Detector</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a portrait image to analyze whether it's <b>Real</b> or <b>AI-Generated</b> with PyTorch ResNet18 ✨</p>", unsafe_allow_html=True)
st.write("")

# File Uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='📸 Uploaded Portrait', use_column_width=True)
        
        with st.spinner("🔍 Running neural network analysis..."):
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            classes = ["Fake", "Real"]
            result_class = classes[predicted.item()]
            conf_score = confidence.item() * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        if result_class == "Real":
            st.success(f"🌸 **Result: Authentic / Real Face** \n\nConfidence Score: **{conf_score:.2f}%**")
        else:
            st.error(f"⚠️ **Result: Deepfake / AI-Generated** \n\nConfidence Score: **{conf_score:.2f}%**")
