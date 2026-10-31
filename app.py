import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

MODEL_URL = "https://drive.google.com/uc?export=download&id=1PUxOsAbMjrb8H5zZmRksWTcgzFzOiIfA"
MODEL_PATH = "deepfake_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Downloading model weights... Please wait."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Download failed: {e}")
                
    # Initialize architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Error loading model weights: {e}. The downloaded file might be corrupted or an HTML page.")
        
    model.eval()
    return model

model = load_model()

# Preprocessing & App UI
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🛡️ Deepfake Face Detection System")
st.write("Upload a facial image to check whether it is **Real** or a **Deepfake** using your trained PyTorch ResNet18 model.")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🔍 Analyzing image...")
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    classes = ["Fake", "Real"]
    result_class = classes[predicted.item()]
    conf_score = confidence.item() * 100
    
    st.markdown("---")
    if result_class == "Real":
        st.success(f"✅ Result: **Real Face** (Confidence: {conf_score:.2f}%)")
    else:
        st.error(f"🚨 Result: **Deepfake / AI-Generated** (Confidence: {conf_score:.2f}%)")
