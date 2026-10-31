import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
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

FILE_ID = "1PUxOsAbMjrb8H5zZmRksWTcgzFzOiIfA"
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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.markdown("<h1>🛡️ Deepfake Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown("<p>Advanced PyTorch ResNet18 biometric verification framework for real-time artifact detection.</p>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Target Image Analysis', use_column_width=True)
        
        with st.spinner("🔍 Executing inference pass..."):
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            classes = ["Real", "Fake"]
            result_class = classes[predicted.item()]
            conf_score = confidence.item() * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        if result_class == "Real":
            st.success(f"✦ **Verification Status: REAL** \n\nConfidence: **{conf_score:.2f}%**")
        else:
            st.error(f"✦ **Verification Status: FAKE** \n\nConfidence: **{conf_score:.2f}%**")
