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

st.set_page_config(layout="wide", page_title="Adaptive Driver Monitoring", initial_sidebar_state="collapsed")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Make metrics more prominent */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Alert boxes */
    .big-alert {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
    }
    
    .alert-danger {
        background-color: #ff4444;
        color: white;
    }
    
    .alert-warning {
        background-color: #ffaa00;
        color: white;
    }
    
    .alert-safe {
        background-color: #00cc66;
        color: white;
    }
    
    /* Compact info boxes */
    .info-box {
        background-color: #1e1e1e;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        border-left: 3px solid #4444ff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load all models and return them as a dictionary."""
    with st.spinner("🔄 Loading AI models..."):
        models = {
            "gaze": GazeTracker(model_path="gaze_tracker_endterm.pth"),
            "road": RoadContext(),
            "fusion": FusionEngine()
        }
    return models

# --- Visualization ---
def draw_gaze_vectors(image: Image.Image, pred_pitch, pred_yaw, gt_pitch=None, gt_yaw=None, length=150, thickness=3):
    """Draw gaze direction vectors on the face image."""
    image_cv = np.array(image.copy())
    h, w = image_cv.shape[:2]
    center = (w // 2, h // 2)
    
    # Predicted gaze (RED)
    dx_pred = int(length * np.sin(pred_yaw))
    dy_pred = int(-length * np.sin(pred_pitch))
    end_pred = (center[0] + dx_pred, center[1] + dy_pred)
    cv2.arrowedLine(image_cv, center, end_pred, (255, 0, 0), thickness, tipLength=0.3)
    
    # Ground truth (GREEN) - if available
    if gt_pitch is not None and gt_yaw is not None:
        dx_gt = int(length * np.sin(gt_yaw))
        dy_gt = int(-length * np.sin(gt_pitch))
        end_gt = (center[0] + dx_gt, center[1] + dy_gt)
        cv2.arrowedLine(image_cv, center, end_gt, (0, 255, 0), thickness, tipLength=0.3)
    
    return Image.fromarray(image_cv)

# --- Helper to find test scenarios ---
def find_test_scenarios(data_dir, dataset_label=""):
    """Find test scenarios in a data directory.
    
    Args:
        data_dir: Directory containing processed test data
        dataset_label: Optional label to prefix scenario names (e.g., "DashGaze", "Brain4Cars")
    
    Returns:
        Dictionary mapping scenario names to (driver_path, road_path, camera_path, gt_path)
    """
    scenarios = {}
    gt_files = sorted(glob.glob(os.path.join(data_dir, "*_gt.json")))
    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace("_gt.json", "")
        driver_path = os.path.join(data_dir, f"{base_name}_driver.jpg")
        road_path = os.path.join(data_dir, f"{base_name}_road.jpg")
        camera_path = os.path.join(data_dir, f"{base_name}_camera.jpg")
        if os.path.exists(driver_path) and os.path.exists(road_path):
            if dataset_label:
                scenario_name = f"[{dataset_label}] {base_name}"
            else:
                scenario_name = f"{base_name}"
            scenarios[scenario_name] = (driver_path, road_path, camera_path, gt_path)
    return scenarios

# --- Load models first (before any UI) ---
models = load_models()

# --- Main App ---
st.title("🚗 Driver Monitoring System")

# Sidebar for global settings
with st.sidebar:
    st.markdown("**🔧 Settings**")
    show_images = st.checkbox("Show camera feeds", value=True)
    show_details = st.checkbox("Show technical details", value=False)

# --- Load scenarios from both datasets ---
DASHGAZE_DIR = os.path.join("test_data", "dashgaze_processed")
BRAIN4CARS_DIR = os.path.join("test_data", "brain4cars_processed")

dashgaze_scenarios = {}
brain4cars_scenarios = {}

if os.path.exists(DASHGAZE_DIR):
    dashgaze_scenarios = find_test_scenarios(DASHGAZE_DIR, "")

if os.path.exists(BRAIN4CARS_DIR):
    brain4cars_scenarios = find_test_scenarios(BRAIN4CARS_DIR, "")

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["📹 DashGaze Dataset", "🚗 Brain4Cars Dataset", "📤 Upload Custom"])

def process_and_display(driver_image, road_image, camera_image, ground_truth, show_images, show_details):
    """Process images and display results."""
    # Placeholder for status (will be filled after processing)
    status_placeholder = st.empty()
    
    # Process gaze
    raw_pitch, raw_yaw = models["gaze"].predict_gaze(driver_image)
    pred_pitch = raw_pitch
    pred_yaw = raw_yaw
    gaze_zone = models["fusion"].classify_gaze_zone(pred_pitch, pred_yaw)
    
    # Process road
    annotated_road_image, road_objects = models["road"].detect_objects(road_image)
        
    # Get ground truth
    gt_azimuth_deg = ground_truth.get('azimuth_deg')
    gt_elevation_deg = ground_truth.get('elevation_deg')
    gt_yaw, gt_pitch = None, None
    if gt_azimuth_deg is not None and gt_elevation_deg is not None:
        gt_yaw = np.deg2rad(gt_azimuth_deg)
        gt_pitch = np.deg2rad(gt_elevation_deg)

    # --- ALERT SECTION (Display at top using placeholder) ---
    # Determine alert level
    assessment = models["fusion"].assess_driver_state(gaze_zone, road_objects)
    
    with status_placeholder.container():
        if "WARNING" in assessment:
            st.markdown(f'<div class="big-alert alert-danger">⚠️ {assessment}</div>', unsafe_allow_html=True)
        elif "CAUTION" in assessment:
            st.markdown(f'<div class="big-alert alert-warning">⚡ {assessment}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="big-alert alert-safe">✅ {assessment}</div>', unsafe_allow_html=True)
    
    # --- KEY METRICS ---
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        # Color-code gaze zone
        if gaze_zone == "Road Ahead":
            st.metric("👁️ Gaze Direction", gaze_zone, delta="Safe", delta_color="normal")
        else:
            st.metric("👁️ Gaze Direction", gaze_zone, delta="Off-road", delta_color="inverse")
    
    with col_m2:
        if road_objects:
            st.metric("🚦 Road Objects", f"{len(set(road_objects))} detected", delta=", ".join(set(road_objects))[:30])
        else:
            st.metric("🚦 Road Objects", "None", delta="Clear road")
    
    with col_m3:
        # Show model accuracy if ground truth available, otherwise show maneuver type
        if gt_azimuth_deg is not None:
            error_yaw = abs(np.rad2deg(pred_yaw) - gt_azimuth_deg)
            error_pitch = abs(np.rad2deg(pred_pitch) - gt_elevation_deg)
            avg_error = (error_yaw + error_pitch) / 2
            st.metric("📊 Model Accuracy", f"{avg_error:.1f}° error", 
                     delta=f"Pitch: {error_pitch:.1f}°, Yaw: {error_yaw:.1f}°")
        else:
            # For Brain4Cars, show maneuver type instead
            maneuver_type = ground_truth.get('maneuver', 'Unknown')
            if maneuver_type != 'Unknown':
                maneuver_display = {
                    'lchange': '⬅️ Left Lane Change',
                    'rchange': '➡️ Right Lane Change',
                    'lturn': '↰ Left Turn',
                    'rturn': '↱ Right Turn',
                    'end_action': '⬆️ Straight'
                }.get(maneuver_type, maneuver_type)
                st.metric("🎯 Maneuver", maneuver_display)
            else:
                st.metric("📊 Gaze Angles", f"P: {np.rad2deg(pred_pitch):.1f}°", 
                         delta=f"Y: {np.rad2deg(pred_yaw):.1f}°")

    # --- CAMERA FEEDS (OPTIONAL) ---
    if show_images:
        st.markdown("---")
        st.markdown("## 📷 Camera Feeds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gaze_image = draw_gaze_vectors(driver_image, pred_pitch, pred_yaw, gt_pitch, gt_yaw)
            st.image(gaze_image, caption="Driver Face", use_container_width=True)
        
        with col2:
            st.image(annotated_road_image, caption="Road View", use_container_width=True)
        
        with col3:
            if camera_image:
                st.image(camera_image, caption="Dashboard Camera", use_container_width=True)
            else:
                st.info("No dashboard camera feed")
    
    # --- TECHNICAL DETAILS (COLLAPSIBLE) ---
    if show_details:
        st.markdown("---")
        with st.expander("🔍 Technical Details", expanded=False):
            col_t1, col_t2, col_t3 = st.columns(3)
            
            with col_t1:
                st.markdown("**Gaze Prediction**")
                st.write(f"Pitch: {np.rad2deg(pred_pitch):.2f}°")
                st.write(f"Yaw: {np.rad2deg(pred_yaw):.2f}°")
                st.write(f"Zone: {gaze_zone}")
            
            with col_t2:
                st.markdown("**Ground Truth**")
                if gt_azimuth_deg is not None:
                    st.write(f"Elevation: {gt_elevation_deg:.2f}°")
                    st.write(f"Azimuth: {gt_azimuth_deg:.2f}°")
                    st.write(f"Pitch Error: {error_pitch:.2f}°")
                    st.write(f"Yaw Error: {error_yaw:.2f}°")
                else:
                    maneuver_type = ground_truth.get('maneuver', 'N/A')
                    st.write(f"Maneuver: {maneuver_type}")
                    st.write("No gaze ground truth")
            
            with col_t3:
                st.markdown("**Road Context**")
                if road_objects:
                    for obj in set(road_objects):
                        st.write(f"• {obj}")
                else:
                    st.write("No objects detected")
        
        with st.expander("ℹ️ About This System", expanded=False):
            st.markdown("""
            **Model Architecture:**
            - **Gaze Tracker**: ResNet18 (MpiiFaceGaze) + Linear Calibration
            - **Object Detection**: YOLOv8-nano
            - **Calibration**: Least-squares regression (507 samples)
            
            **Performance:**
            - Pitch Error: 17.6° → 2.8° (84% improvement)
            - Yaw Error: 4.4° → 4.1° (7% improvement)
            
            **Gaze Zones:**
            - Road Ahead: ±8° yaw, ±8° pitch
            - Mirrors: ±10-60° yaw, ±25-20° pitch
            - Down (Phone/Lap): ±12° yaw, 8-20° pitch
            """)

# --- TAB 1: DashGaze Dataset ---
with tab1:
    st.markdown("### 📹 DashGaze Test Scenarios")
    st.markdown("DashGaze dataset with ground truth gaze annotations for accuracy evaluation")
    
    if dashgaze_scenarios:
        scenario_keys = list(dashgaze_scenarios.keys())
        selected_scenario = st.selectbox("Select DashGaze Scenario", scenario_keys, key="dashgaze_select")
        
        if selected_scenario:
            driver_path, road_path, camera_path, gt_path = dashgaze_scenarios[selected_scenario]
            driver_image = Image.open(driver_path)
            road_image = Image.open(road_path)
            camera_image = Image.open(camera_path) if os.path.exists(camera_path) else None
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
            
            process_and_display(driver_image, road_image, camera_image, ground_truth, show_images, show_details)
    else:
        st.warning("⚠️ No DashGaze scenarios found. Please run preprocessing first.")
        st.code("python scripts/preprocess_dashgaze.py")

# --- TAB 2: Brain4Cars Dataset ---
with tab2:
    st.markdown("### 🚗 Brain4Cars Test Scenarios")
    st.markdown("Brain4Cars dataset with driving maneuver annotations (lane changes, turns, straight driving)")
    
    if brain4cars_scenarios:
        scenario_keys = list(brain4cars_scenarios.keys())
        selected_scenario = st.selectbox("Select Brain4Cars Scenario", scenario_keys, key="brain4cars_select")
        
        if selected_scenario:
            driver_path, road_path, camera_path, gt_path = brain4cars_scenarios[selected_scenario]
            driver_image = Image.open(driver_path)
            road_image = Image.open(road_path)
            camera_image = Image.open(camera_path) if os.path.exists(camera_path) else None
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
            
            process_and_display(driver_image, road_image, camera_image, ground_truth, show_images, show_details)
    else:
        st.warning("⚠️ No Brain4Cars scenarios found. Please run preprocessing first.")
        st.code("python scripts/preprocess_brain4cars.py")

# --- TAB 3: Upload Custom ---
with tab3:
    st.markdown("### 📤 Upload Your Own Images")
    st.markdown("Upload a driver face image and road view image for custom testing")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        driver_file = st.file_uploader("Driver Face Image", type=["jpg", "jpeg", "png"], key="driver")
    
    with col_upload2:
        road_file = st.file_uploader("Road View Image", type=["jpg", "jpeg", "png"], key="road")
    
    if driver_file and road_file:
        driver_image = Image.open(driver_file)
        road_image = Image.open(road_file)
        ground_truth = {}  # No ground truth for custom uploads
        
        process_and_display(driver_image, road_image, None, ground_truth, show_images, show_details)