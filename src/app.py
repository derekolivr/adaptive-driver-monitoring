import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import glob
import json

# Import the modules we created
from gaze_tracker import GazeTracker
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
            "road": RoadContext(),
            "fusion": FusionEngine()
        }
    return models

# --- Visualization ---
def draw_gaze_vectors(image: Image.Image, pred_pitch, pred_yaw, gt_pitch=None, gt_yaw=None, length=200, thickness=5):
    """
    Draws the predicted gaze vector (red) and optionally the ground truth gaze vector (green).
    """
    image_cv = np.array(image.convert('RGB'))
    h, w, _ = image_cv.shape
    center_x, center_y = w // 2, h // 2
    
    # Draw Predicted Gaze (Red)
    pred_end_x = int(center_x + length * np.sin(pred_yaw))
    pred_end_y = int(center_y - length * np.sin(pred_pitch))
    cv2.line(image_cv, (center_x, center_y), (pred_end_x, pred_end_y), (0, 0, 255), thickness) # Red in BGR

    # Draw Ground Truth Gaze (Green), if available
    if gt_pitch is not None and gt_yaw is not None:
        gt_end_x = int(center_x + length * np.sin(gt_yaw))
        gt_end_y = int(center_y - length * np.sin(gt_pitch))
        cv2.line(image_cv, (center_x, center_y), (gt_end_x, gt_end_y), (0, 255, 0), thickness) # Green in BGR

    return Image.fromarray(image_cv)

# --- Helper to find test scenarios ---
def find_test_scenarios(data_dir):
    scenarios = {}
    gt_files = sorted(glob.glob(os.path.join(data_dir, "*_gt.json")))
    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace("_gt.json", "")
        driver_path = os.path.join(data_dir, f"{base_name}_driver.jpg")
        road_path = os.path.join(data_dir, f"{base_name}_road.jpg")
        if os.path.exists(driver_path) and os.path.exists(road_path):
            scenario_name = f"Scenario: {base_name}"
            scenarios[scenario_name] = (driver_path, road_path, gt_path)
    return scenarios

# --- Main App ---
st.title("🚗 Adaptive Driver Monitoring System")
st.write("An integrated system for analyzing driver attention and road context.")

# --- Sidebar ---
st.sidebar.title("Test Scenarios")
st.sidebar.write("Select a synchronized scenario from the DashGaze dataset.")

TEST_DATA_DIR = os.path.join("test_data", "dashgaze_processed")
scenarios = find_test_scenarios(TEST_DATA_DIR)
scenario_keys = ["-- Upload Your Own Images --"] + list(scenarios.keys())
selected_scenario_key = st.sidebar.selectbox("Choose a scenario", scenario_keys)

models = load_models()

# Initialize variables
driver_image, road_image, ground_truth = None, None, {}
gaze_zone, road_objects = None, []

# --- Data Loading ---
if selected_scenario_key != "-- Upload Your Own Images --":
    driver_path, road_path, gt_path = scenarios[selected_scenario_key]
    driver_image = Image.open(driver_path)
    road_image = Image.open(road_path)
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
else:
    driver_file = st.sidebar.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"], key="driver")
    road_file = st.sidebar.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"], key="road")
    if driver_file: driver_image = Image.open(driver_file)
    if road_file: road_image = Image.open(road_file)

# --- Model Processing & Display ---
col1, col2 = st.columns(2)

with col1:
    st.header("Driver Gaze Analysis")
    if driver_image:
        # Run gaze prediction
        pred_pitch, pred_yaw = models["gaze"].predict_gaze(driver_image)
        
        # Apply coordinate calibration for DashGaze dataset
        # The model was trained on MpiiFaceGaze, but we're testing on DashGaze
        # These may need different coordinate conventions
        # Note: Negative signs and offsets may need adjustment based on visual inspection
        calibrated_pitch = -pred_pitch  # Flip pitch direction
        calibrated_yaw = pred_yaw  # Keep yaw as is for now
        
        gaze_zone = models["fusion"].classify_gaze_zone(calibrated_pitch, calibrated_yaw)
        
        # Get ground truth gaze for comparison
        gt_azimuth_deg = ground_truth.get('azimuth_deg')
        gt_elevation_deg = ground_truth.get('elevation_deg')
        gt_yaw, gt_pitch = None, None
        if gt_azimuth_deg is not None and gt_elevation_deg is not None:
            # Convert degrees from dataset to radians for our function
            gt_yaw = np.deg2rad(gt_azimuth_deg)
            gt_pitch = np.deg2rad(gt_elevation_deg)
        
        # Draw gaze vectors and display (using calibrated values for visualization)
        gaze_image = draw_gaze_vectors(driver_image, calibrated_pitch, calibrated_yaw, gt_pitch, gt_yaw)
        st.image(gaze_image, caption="Gaze Prediction (Red: Predicted, Green: Ground Truth)", use_container_width=True)
        
        # Display results
        st.subheader("Gaze Zone Classification")
        st.metric("Predicted Gaze Zone", gaze_zone)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Raw Model Output", f"P: {np.rad2deg(pred_pitch):.1f}°, Y: {np.rad2deg(pred_yaw):.1f}°")
        with col_b:
            st.metric("Calibrated Angles", f"P: {np.rad2deg(calibrated_pitch):.1f}°, Y: {np.rad2deg(calibrated_yaw):.1f}°")
        
        # Show ground truth comparison if available
        if gt_azimuth_deg is not None:
            st.subheader("Ground Truth Comparison")
            st.markdown(f"**Dataset Gaze:** Azimuth: `{gt_azimuth_deg:.2f}°`, Elevation: `{gt_elevation_deg:.2f}°`")
            
            error_yaw = abs(np.rad2deg(calibrated_yaw) - gt_azimuth_deg)
            error_pitch = abs(np.rad2deg(calibrated_pitch) - gt_elevation_deg)
            
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Azimuth Error", f"{error_yaw:.2f}°")
            with col_d:
                st.metric("Elevation Error", f"{error_pitch:.2f}°")
            
            # Add explanation
            with st.expander("ℹ️ About the errors"):
                st.markdown("""
                **Why are there errors?**
                
                1. **Domain Mismatch**: The model was trained on MpiiFaceGaze (lab environment) but tested on DashGaze (real driving).
                2. **Different Camera Setup**: Camera position and angle differ between datasets.
                3. **Coordinate System**: The two datasets may use different angle conventions.
                
                **Expected Performance**: 5-15° error is typical for cross-dataset gaze estimation. 
                Errors >20° suggest significant domain shift or coordinate system mismatch.
                """)
    else:
        st.info("Upload a driver image or select a scenario.")

with col2:
    st.header("Road Context")
    if road_image:
        annotated_road_image, road_objects = models["road"].detect_objects(road_image)
        st.image(annotated_road_image, caption="Road Object Detection", use_column_width=True)
        if road_objects:
            st.info(f"**Detected Objects:** {', '.join(set(road_objects))}")
        else:
            st.info("No relevant objects detected.")
    else:
        st.info("Upload a road image or select a scenario.")

# --- Fusion Engine Assessment ---
if driver_image and road_image:
    st.header("Overall Driver Attention Assessment")
    assessment = models["fusion"].assess_driver_state(gaze_zone, road_objects)
    
    if "WARNING" in assessment or "CAUTION" in assessment:
        st.warning(f"**Status:** {assessment}")
    else:
        st.success(f"**Status:** {assessment}")