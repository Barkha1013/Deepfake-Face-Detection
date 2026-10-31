import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# 1. Load and Cache the ResNet18 Model
@st.cache_resource
def load_model():
    # Initialize ResNet18 architecture matching your training notebook
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load your trained weights file
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 2. Define Image Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Streamlit Web App Interface Layout
st.title("🛡️ Deepfake Face Detection System")
st.write("Upload a facial image to check whether it is **Real** or a **Deepfake** using your trained PyTorch ResNet18 model.")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🔍 Analyzing image...")
    
    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Class mapping based on your training folder names
    # Note: verify order of dataset.classes from your training notebook
    classes = ["Fake", "Real"]
    result_class = classes[predicted.item()]
    conf_score = confidence.item() * 100
    
    # Display results
    st.markdown("---")
    if result_class == "Real":
        st.success(f"✅ Result: **Real Face** (Confidence: {conf_score:.2f}%)")
    else:
        st.error(f"🚨 Result: **Deepfake / AI-Generated** (Confidence: {conf_score:.2f}%)")
