import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os # Import the os module

# Import the modules we created (paths updated for new location)
from gaze_tracker import GazeTracker
from distraction_classifier import DistractionClassifier
from road_context import RoadContext
from fusion_engine import FusionEngine

st.set_page_config(layout="wide", page_title="Adaptive Driver Monitoring")

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load all models and return them as a dictionary."""
    with st.spinner("Loading models... This may take a moment."):
        models = {
            "gaze": GazeTracker(model_path="gaze_tracker_endterm.pth"),
            "distraction": DistractionClassifier(model_path=os.path.join("Midterm", "driver_distraction_model_vgg.h5")),
            "road": RoadContext(),
            "fusion": FusionEngine()
        }
    return models

# --- Visualization ---
def draw_gaze(image: Image.Image, pitch, yaw, length=200, thickness=5):
    """Draws the gaze vector on the image."""
    image_cv = np.array(image.convert('RGB'))
    h, w, _ = image_cv.shape
    center_x, center_y = w // 2, h // 2
    
    end_x = int(center_x + length * np.sin(yaw))
    end_y = int(center_y - length * np.sin(pitch))

    cv2.line(image_cv, (center_x, center_y), (end_x, end_y), (0, 0, 255), thickness)
    return Image.fromarray(image_cv)

# --- Main App ---
st.title("🚗 Adaptive Driver Monitoring System")
st.write("An integrated system using Gaze Tracking, Distraction Classification, and Road Context Analysis.")

models = load_models()

col1, col2 = st.columns(2)

# --- Driver Analysis Column ---
with col1:
    st.header("Driver Analysis")
    driver_image_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"], key="driver")

    if driver_image_file:
        driver_image = Image.open(driver_image_file)
        st.image(driver_image, caption="Original Driver Image", use_column_width=True)

        # 1. Gaze Tracker
        pitch, yaw = models["gaze"].predict_gaze(driver_image)
        gaze_image = draw_gaze(driver_image, pitch, yaw)
        st.image(gaze_image, caption="Gaze Prediction", use_column_width=True)
        gaze_zone = models["fusion"].classify_gaze_zone(pitch, yaw)
        st.info(f"**Gaze Zone:** {gaze_zone}")
        
        # 2. Distraction Classifier
        distraction_label, confidence = models["distraction"].classify_distraction(driver_image)
        st.info(f"**Distraction:** {distraction_label} ({confidence:.2%})")

# --- Road Analysis Column ---
with col2:
    st.header("Road Context")
    road_image_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"], key="road")
    
    road_objects = []
    if road_image_file:
        road_image = Image.open(road_image_file)
        
        annotated_road_image, road_objects = models["road"].detect_objects(road_image)
        st.image(annotated_road_image, caption="Road Object Detection", use_column_width=True)
        
        if road_objects:
            st.info(f"**Detected Objects:** {', '.join(set(road_objects))}")
        else:
            st.info("No relevant objects detected in the road view.")

# --- Fusion Engine Assessment ---
if driver_image_file and road_image_file:
    st.header(" 종합 상황 분석 (Overall Situation Analysis)")
    
    assessment = models["fusion"].assess_driver_state(gaze_zone, distraction_label, road_objects)
    
    if "ALERT" in assessment:
        st.error(f"**Status:** {assessment}")
    elif "WARNING" in assessment or "CAUTION" in assessment:
        st.warning(f"**Status:** {assessment}")
    else:
        st.success(f"**Status:** {assessment}")