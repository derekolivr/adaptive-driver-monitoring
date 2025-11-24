import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# To make sure the app can find the model definition
import sys
sys.path.append('Driver_Gaze_Map')
from model import get_gaze_model

# --- Model Loading ---
@st.cache_resource  # Caches the model so it doesn't reload on every interaction
def load_model(model_path="gaze_tracker_best.pth"):
    """Loads the gaze tracker model and its weights."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Make sure it's in the root directory.")
        return None

    # Set device
    device = torch.device("cpu") # Streamlit cloud runs on CPU, this is safer
    
    # Load model architecture and weights
    model = get_gaze_model()
    # The map_location ensures the model loads correctly even if trained on a GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Prediction and Visualization ---
def predict_gaze(model, image: Image.Image):
    """Runs the model to predict gaze angles from a PIL image."""
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    pitch, yaw = output.numpy()[0]
    return pitch, yaw

def draw_gaze(image: Image.Image, pitch, yaw, length=150, thickness=3):
    """Draws the gaze vector on the image."""
    image_cv = np.array(image.convert('RGB'))
    h, w, _ = image_cv.shape
    center_x, center_y = w // 2, h // 2

    # A positive yaw is "right", negative is "left"
    # A negative pitch is "up", positive is "down"
    
    # Corrected calculation for the endpoint of the gaze vector
    end_x = int(center_x + length * np.sin(yaw))
    end_y = int(center_y - length * np.sin(pitch))

    # Draw the gaze vector as a red line from the center of the image
    cv2.line(image_cv, (center_x, center_y), (end_x, end_y), (0, 0, 255), thickness)
    return Image.fromarray(image_cv)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("👁️ Driver Gaze Tracker Demo")
st.write("Upload an image of a driver's face to predict their gaze direction.")

# Load the model
model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, use_column_width=True)

        # Make prediction
        pitch, yaw = predict_gaze(model, image)
        pitch_deg, yaw_deg = np.rad2deg(pitch), np.rad2deg(yaw)
        
        # Draw gaze on the image
        gaze_image = draw_gaze(image, pitch, yaw)

        with col2:
            st.header("Predicted Gaze")
            st.image(gaze_image, use_column_width=True)

        st.subheader("Prediction Results:")
        st.metric(label="Pitch (Vertical Angle)", value=f"{pitch_deg:.2f}°", delta="Negative is Up, Positive is Down")
        st.metric(label="Yaw (Horizontal Angle)", value=f"{yaw_deg:.2f}°", delta="Negative is Left, Positive is Right")
else:
    st.warning("Could not load the model. Please ensure `gaze_tracker_best.pth` is in the same directory as `app.py`.")