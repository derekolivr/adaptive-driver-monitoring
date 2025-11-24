import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os # Import the os module
import glob

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

# --- Helper to find test scenarios ---
def find_test_scenarios(data_dir):
    scenarios = {}
    driver_files = sorted(glob.glob(os.path.join(data_dir, "*_driver.jpg")))
    for driver_path in driver_files:
        base_name = os.path.basename(driver_path).replace("_driver.jpg", "")
        road_path = os.path.join(data_dir, f"{base_name}_road.jpg")
        if os.path.exists(road_path):
            scenario_name = f"Scenario: {base_name}"
            scenarios[scenario_name] = (driver_path, road_path)
    return scenarios

# --- Main App ---
st.title("🚗 Adaptive Driver Monitoring System")
st.write("An integrated system using Gaze Tracking, Distraction Classification, and Road Context Analysis.")

# --- Sidebar for Test Scenarios ---
st.sidebar.title("Test Scenarios")
st.sidebar.write("Select a pre-defined, synchronized scenario from the DashGaze dataset.")

TEST_DATA_DIR = os.path.join("test_data", "dashgaze_processed")
scenarios = find_test_scenarios(TEST_DATA_DIR)

# Prepend the file uploader option to the scenarios dict
scenario_keys = ["-- Upload Your Own Images --"] + list(scenarios.keys())
selected_scenario_key = st.sidebar.selectbox("Choose a scenario", scenario_keys)

models = load_models()

col1, col2 = st.columns(2)

# Initialize image variables
driver_image = None
road_image = None

# --- Driver Analysis Column ---
with col1:
    st.header("Driver Analysis")
    if selected_scenario_key != "-- Upload Your Own Images --":
        driver_path, _ = scenarios[selected_scenario_key]
        driver_image = Image.open(driver_path)
    else:
        driver_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"], key="driver")
        if driver_file:
            driver_image = Image.open(driver_file)

    if driver_image:
        st.image(driver_image, caption="Driver Image", use_column_width=True)
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
    road_objects = []
    if selected_scenario_key != "-- Upload Your Own Images --":
        _, road_path = scenarios[selected_scenario_key]
        road_image = Image.open(road_path)
    else:
        road_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"], key="road")
        if road_file:
            road_image = Image.open(road_file)
    
    if road_image:
        annotated_road_image, road_objects = models["road"].detect_objects(road_image)
        st.image(annotated_road_image, caption="Road Object Detection", use_column_width=True)
        
        if road_objects:
            st.info(f"**Detected Objects:** {', '.join(set(road_objects))}")
        else:
            st.info("No relevant objects detected in the road view.")

# --- Fusion Engine Assessment ---
if driver_image and road_image:
    st.header(" 종합 상황 분석 (Overall Situation Analysis)")
    
    # These variables are only defined if a driver image has been processed
    assessment = models["fusion"].assess_driver_state(gaze_zone, distraction_label, road_objects)
    
    if "ALERT" in assessment:
        st.error(f"**Status:** {assessment}")
    elif "WARNING" in assessment or "CAUTION" in assessment:
        st.warning(f"**Status:** {assessment}")
    else:
        st.success(f"**Status:** {assessment}")