import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import glob
import json
from typing import Dict

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
        background-color: #ffd700;
        color: #000000;
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
def draw_gaze_vectors(image: Image.Image, pred_pitch, pred_yaw, gt_pitch=None, gt_yaw=None, length=200, thickness=5):
    """Draw gaze direction vectors on the face image."""
    image_cv = np.array(image.copy())
    h, w = image_cv.shape[:2]
    center = (w // 2, h // 2)
    
    # Predicted gaze (RED)
    # Note: In image coordinates, Y increases downward, so we flip pitch sign
    dx_pred = int(length * np.sin(pred_yaw))
    dy_pred = int(length * np.sin(pred_pitch))  # Flipped sign for correct visualization
    end_pred = (center[0] + dx_pred, center[1] + dy_pred)
    cv2.arrowedLine(image_cv, center, end_pred, (255, 0, 0), thickness, tipLength=0.3)
    
    # Ground truth (GREEN) - if available
    if gt_pitch is not None and gt_yaw is not None:
        dx_gt = int(length * np.sin(gt_yaw))
        dy_gt = int(length * np.sin(gt_pitch))  # Flipped sign for correct visualization
        end_gt = (center[0] + dx_gt, center[1] + dy_gt)
        cv2.arrowedLine(image_cv, center, end_gt, (0, 255, 0), thickness, tipLength=0.3)
    
    return Image.fromarray(image_cv)

# --- Helper to find test scenarios ---
def find_brain4cars_scenarios(data_dir: str) -> Dict:
    """Finds Brain4Cars scenarios, which are organized as directories of frames."""
    scenarios = {}
    if not os.path.exists(data_dir):
        return scenarios
        
    sample_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*/")) if os.path.isdir(d)])
    
    for sample_dir in sample_dirs:
        gt_path = os.path.join(sample_dir, "_gt.json")
        if os.path.exists(gt_path):
            base_name = os.path.basename(os.path.normpath(sample_dir))
            scenarios[base_name] = sample_dir
            
    return scenarios

def find_dashgaze_scenarios(data_dir: str) -> Dict:
    """Finds DashGaze scenarios, which are organized as individual image files."""
    scenarios = {}
    if not os.path.exists(data_dir):
        return scenarios

    gt_files = sorted(glob.glob(os.path.join(data_dir, "*_gt.json")))
    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace("_gt.json", "")
        driver_path = os.path.join(data_dir, f"{base_name}_driver.jpg")
        road_path = os.path.join(data_dir, f"{base_name}_road.jpg")
        camera_path = os.path.join(data_dir, f"{base_name}_camera.jpg")
        if os.path.exists(driver_path) and os.path.exists(road_path):
            scenarios[base_name] = (driver_path, road_path, camera_path, gt_path)
            
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
    dashgaze_scenarios = find_dashgaze_scenarios(DASHGAZE_DIR)

if os.path.exists(BRAIN4CARS_DIR):
    brain4cars_scenarios = find_brain4cars_scenarios(BRAIN4CARS_DIR)

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["📹 DashGaze Dataset", "🚗 Brain4Cars Dataset", "📤 Upload Custom"])

def process_and_display(driver_image, road_image, ground_truth, show_images, show_details):
    """Process images and display results."""
    # Placeholder for status (will be filled after processing)
    status_placeholder = st.empty()
    
    # Determine if we should apply calibration (only for DashGaze, not Brain4Cars)
    is_brain4cars = ground_truth.get('source') == 'Brain4Cars'
    apply_calibration = not is_brain4cars
    
    # Process gaze
    pred_pitch, pred_yaw = models["gaze"].predict_gaze(driver_image, apply_calibration=apply_calibration)
    
    # Classify gaze zone
    gaze_zone = models["fusion"].classify_gaze_zone(pred_pitch, pred_yaw)
    
    # Debug output for Brain4Cars to verify values
    if is_brain4cars:
        with st.sidebar.expander("🔍 Debug - Gaze Values", expanded=show_details):
            st.write(f"**Final values:**")
            st.write(f"Pitch: {np.rad2deg(pred_pitch):.2f}° ({pred_pitch:.4f} rad)")
            st.write(f"Yaw: {np.rad2deg(pred_yaw):.2f}° ({pred_yaw:.4f} rad)")
            st.write(f"**Gaze Zone:** {gaze_zone}")
            st.write(f"**Note:** If values are clustered, the model may need different interpretation")
    
    # Process road
    annotated_road_image, road_objects = models["road"].detect_objects(road_image)
        
    # Get ground truth and Brain4Cars context
    gt_azimuth_deg = ground_truth.get('azimuth_deg')
    gt_elevation_deg = ground_truth.get('elevation_deg')
    maneuver = ground_truth.get('maneuver')
    is_near_intersection = ground_truth.get('is_near_intersection', False)

    gt_yaw, gt_pitch = None, None
    if gt_azimuth_deg is not None and gt_elevation_deg is not None:
        gt_yaw = np.deg2rad(gt_azimuth_deg)
        gt_pitch = np.deg2rad(gt_elevation_deg)

    # --- ALERT SECTION (Display at top using placeholder) ---
    # Determine alert level
    assessment = models["fusion"].assess_driver_state(
        gaze_zone, 
        road_objects, 
        maneuver=maneuver, 
        is_near_intersection=is_near_intersection
    )
    
    with status_placeholder.container():
        if "CAUTION" in assessment:
            st.markdown(f'<div class="big-alert alert-warning">{assessment}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="big-alert alert-safe">{assessment}</div>', unsafe_allow_html=True)
    
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
                    'lchange': '⬅️ Left Change',
                    'rchange': '➡️ Right Change',
                    'lturn': '↰ Left Turn',
                    'rturn': '↱ Right Turn',
                    'end_action': '⬆️ Straight'
                }.get(maneuver_type, maneuver_type)
                st.metric("🎯 Maneuver", maneuver_display, 
                         delta="At Intersection" if is_near_intersection else "")
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
            # Vehicle Context section
            st.markdown("##### 🚗 Vehicle Context")
            lane_info = ground_truth.get('lane_info', {})
            
            # Create a styled info box
            intersection_status = "Clear" if not is_near_intersection else "Approaching"
            current_lane = lane_info.get('current_lane', 0)
            total_lanes = lane_info.get('total_lanes', 0)
            
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 1rem; border-radius: 0.5rem; color: #4db8ff;">
                <p style="margin: 0.5rem 0;"><strong>• Intersection Status:</strong> {intersection_status}</p>
                <p style="margin: 0.5rem 0;"><strong>• Lane Position:</strong> {current_lane} of {total_lanes}</p>
            </div>
            """, unsafe_allow_html=True)
    
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
                    st.write(f"Intersection: {is_near_intersection}")
            
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
            # DashGaze still uses single files
            driver_path, road_path, camera_path, gt_path = dashgaze_scenarios[selected_scenario]
            driver_image = Image.open(driver_path)
            road_image = Image.open(road_path)
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
            
            process_and_display(driver_image, road_image, ground_truth, show_images, show_details)
    else:
        st.warning("⚠️ No DashGaze scenarios found. Please run preprocessing first.")
        st.code("python scripts/preprocess_dashgaze.py")

# --- TAB 2: Brain4Cars Dataset ---
with tab2:
    st.markdown("### 🚗 Brain4Cars Test Scenarios")
    st.markdown("Brain4Cars dataset with driving maneuver annotations (lane changes, turns, straight driving)")
    
    if brain4cars_scenarios:
        # Group scenarios by maneuver type
        maneuvers = {}
        for scenario_name, sample_dir in brain4cars_scenarios.items():
            gt_path = os.path.join(sample_dir, "_gt.json")
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                maneuver_type = gt_data.get('maneuver', 'unknown')
                if maneuver_type not in maneuvers:
                    maneuvers[maneuver_type] = []
                maneuvers[maneuver_type].append((scenario_name, sample_dir, gt_data))
        
        # Display maneuver selector
        maneuver_display = {
            'lchange': '⬅️ Left Lane Change',
            'rchange': '➡️ Right Lane Change',
            'lturn': '↰ Left Turn',
            'rturn': '↱ Right Turn',
            'end_action': '⬆️ Straight Driving'
        }
        
        maneuver_keys = sorted([m for m in maneuvers.keys() if m in maneuver_display])
        if maneuver_keys:
            selected_maneuver = st.selectbox(
                "Select Maneuver Type", 
                maneuver_keys,
                format_func=lambda x: maneuver_display.get(x, x),
                key="brain4cars_maneuver"
            )
            
            # Collect all frames from all samples of this maneuver
            all_frames = []
            for scenario_name, sample_dir, gt_data in maneuvers[selected_maneuver]:
                num_frames = gt_data.get("num_frames", 1)
                for frame_idx in range(num_frames):
                    driver_path = os.path.join(sample_dir, f"driver_{frame_idx:02d}.jpg")
                    road_path = os.path.join(sample_dir, f"road_{frame_idx:02d}.jpg")
                    if os.path.exists(driver_path) and os.path.exists(road_path):
                        all_frames.append({
                            'driver_path': driver_path,
                            'road_path': road_path,
                            'ground_truth': gt_data,
                            'scenario': scenario_name,
                            'frame_idx': frame_idx
                        })
            
            if all_frames:
                # Frame slider across all samples
                total_frames = len(all_frames)
                frame_idx = st.slider(
                    f"Frame Sequence ({total_frames} frames across {len(maneuvers[selected_maneuver])} samples)",
                    0, 
                    total_frames - 1, 
                    0,
                    key="brain4cars_frame"
                )
                
                # Get the selected frame
                selected_frame = all_frames[frame_idx]
                driver_image = Image.open(selected_frame['driver_path'])
                road_image = Image.open(selected_frame['road_path'])
                ground_truth = selected_frame['ground_truth']
                
                # Show which sample and frame we're viewing
                st.caption(f"📹 Sample: {selected_frame['scenario']} | Frame: {selected_frame['frame_idx']:02d}")
                
                process_and_display(driver_image, road_image, ground_truth, show_images, show_details)
            else:
                st.error("No frame images found for this maneuver type.")
        else:
            st.warning("No valid maneuvers found in Brain4Cars data.")

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
        
        process_and_display(driver_image, road_image, ground_truth, show_images, show_details)