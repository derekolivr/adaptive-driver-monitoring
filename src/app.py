import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import glob
import json
import re
from typing import Dict
from collections import defaultdict

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
def draw_gaze_vectors(image: Image.Image, pred_pitch, pred_yaw, gt_pitch=None, gt_yaw=None, length=120, thickness=4, flip_horizontal=False, flip_both_axes=False):
    """
    Draw gaze direction vectors on the face image.
    
    Coordinate system:
    - Yaw: positive = right, negative = left (horizontal)
    - Pitch: positive = down, negative = up (vertical in driving context)
    - Image coordinates: x increases right, y increases downward
    
    Vector calculation calibrated to DashGaze ground truth:
    - Typical gaze: ±5° yaw, -5° to +15° pitch
    - Scale factor adjusted to make vectors visible but proportional
    
    Args:
        flip_horizontal: If True, negates yaw to flip horizontal direction (for Brain4Cars)
        flip_both_axes: If True, negates both yaw and pitch (for Driver 8 with opposite coordinate system)
    """
    # Convert PIL RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]
    center = (w // 2, h // 2)
    
    # Apply transformations based on coordinate system
    if flip_both_axes:
        # Driver 8: flip both axes (complete coordinate system inversion)
        yaw_pred = -pred_yaw
        pitch_pred = -pred_pitch
    elif flip_horizontal:
        # Other Brain4Cars: flip horizontal only
        yaw_pred = -pred_yaw
        pitch_pred = pred_pitch
    else:
        # DashGaze: no flip
        yaw_pred = pred_yaw
        pitch_pred = pred_pitch
    
    # Predicted gaze (RED)
    # Scale factor calibrated to ground truth observations
    # DashGaze GT shows small angles (±5° typical), so we scale for visibility
    scale = length / np.deg2rad(20)  # 20° = full length
    dx_pred = int(scale * yaw_pred)
    dy_pred = int(scale * pitch_pred)  # Use transformed pitch
    end_pred = (center[0] + dx_pred, center[1] + dy_pred)
    
    # Add text label with angle values
    cv2.arrowedLine(image_cv, center, end_pred, (0, 0, 255), thickness, tipLength=0.3)  # Red in BGR
    label = f"P:{np.rad2deg(pitch_pred):.0f}° Y:{np.rad2deg(yaw_pred):.0f}°"
    cv2.putText(image_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Ground truth (GREEN) - if available
    if gt_pitch is not None and gt_yaw is not None:
        if flip_both_axes:
            yaw_gt = -gt_yaw
            pitch_gt = -gt_pitch
        elif flip_horizontal:
            yaw_gt = -gt_yaw
            pitch_gt = gt_pitch
        else:
            yaw_gt = gt_yaw
            pitch_gt = gt_pitch
        
        scale = length / np.deg2rad(20)  # Same scaling as prediction
        dx_gt = int(scale * yaw_gt)
        dy_gt = int(scale * pitch_gt)
        end_gt = (center[0] + dx_gt, center[1] + dy_gt)
        cv2.arrowedLine(image_cv, center, end_gt, (0, 255, 0), thickness, tipLength=0.3)  # Green in BGR
        label_gt = f"GT P:{np.rad2deg(pitch_gt):.0f}° Y:{np.rad2deg(yaw_gt):.0f}°"
        cv2.putText(image_cv, label_gt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Convert BGR back to RGB for PIL
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

# --- Helper to find test scenarios ---
def extract_driver_date(clip_id: str) -> str:
    """Extract date (YYYYMMDD) from clip_id like '20141019_091035_1106_1256'."""
    match = re.match(r'(\d{8})_', clip_id)
    if match:
        return match.group(1)
    return "unknown"

def extract_time_from_clip_id(clip_id: str) -> int:
    """Extract time (HHMMSS) from clip_id like '20141019_091035_1106_1256'."""
    match = re.match(r'\d{8}_(\d{6})_', clip_id)
    if match:
        return int(match.group(1))
    return 0

def get_actual_driver_id(clip_id: str) -> tuple:
    """
    Map clip_id to actual driver ID based on user-identified driver groups.
    Returns (driver_id, driver_display_name) tuple.
    
    Rules:
    - Dates 20141025, 20141101, 20141102 are the same person (Driver 2)
    - Date 20141019 has multiple drivers split by time of day:
      * Morning (< 12:00) = Driver 1A
      * Afternoon (>= 12:00) = Driver 1B
    - Other dates are separate drivers
    """
    date = extract_driver_date(clip_id)
    
    # Merge dates 20141025, 20141101, 20141102 into one driver
    if date in ["20141025", "20141101", "20141102"]:
        return ("Driver_2", "Driver 2 (20141025, 20141101, 20141102)")
    
    # Split 20141019 by time of day
    if date == "20141019":
        time = extract_time_from_clip_id(clip_id)
        hour = time // 10000
        if hour < 12:
            return ("Driver_1A", "Driver 1A (20141019 - Morning)")
        else:
            return ("Driver_1B", "Driver 1B (20141019 - Afternoon)")
    
    # Map other dates to driver numbers
    date_to_driver = {
        "20141105": ("Driver_3", "Driver 3 (20141105)"),
        "20141115": ("Driver_4", "Driver 4 (20141115)"),
        "20141116": ("Driver_5", "Driver 5 (20141116)"),
        "20141123": ("Driver_6", "Driver 6 (20141123)"),
        "20141126": ("Driver_7", "Driver 7 (20141126)"),
        "20141220": ("Driver_8", "Driver 8 (20141220)"),
    }
    
    if date in date_to_driver:
        return date_to_driver[date]
    
    # Fallback for unknown dates
    if date != "unknown":
        return (f"Driver_{date}", f"Driver ({date})")
    
    return ("Driver_Unknown", "Driver (Unknown)")

def find_brain4cars_scenarios(data_dir: str) -> Dict:
    """Finds Brain4Cars scenarios, which are organized as directories of frames."""
    scenarios = {}
    if not os.path.exists(data_dir):
        return scenarios
    
    # First, find directories with _gt.json inside
    sample_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*/")) if os.path.isdir(d)])
    
    for sample_dir in sample_dirs:
        gt_path = os.path.join(sample_dir, "_gt.json")
        if os.path.exists(gt_path):
            base_name = os.path.basename(os.path.normpath(sample_dir))
            scenarios[base_name] = sample_dir
    
    # Also check for _gt.json files at root level (they correspond to directories)
    root_gt_files = glob.glob(os.path.join(data_dir, "*_gt.json"))
    for gt_file in root_gt_files:
        # Extract base name (e.g., "end_action_20141019_091035_1106_1256" from "end_action_20141019_091035_1106_1256_gt.json")
        base_name = os.path.basename(gt_file).replace("_gt.json", "")
        # Check if corresponding directory exists
        sample_dir = os.path.join(data_dir, base_name)
        if os.path.isdir(sample_dir) and base_name not in scenarios:
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
    
    # Determine driver-specific flip requirement
    # All Brain4Cars drivers need horizontal flip
    needs_horizontal_flip = is_brain4cars
    needs_both_axes_flip = False
    
    # Process gaze
    pred_pitch, pred_yaw = models["gaze"].predict_gaze(driver_image, apply_calibration=apply_calibration)
    
    # Classify gaze zone
    gaze_zone = models["fusion"].classify_gaze_zone(pred_pitch, pred_yaw)
    
    # Debug output for Brain4Cars to verify values
    if is_brain4cars:
        with st.sidebar.expander("🔍 Debug - Gaze Values", expanded=show_details):
            # Show raw model outputs
            raw_pitch = models["gaze"].last_raw_pitch
            raw_yaw = models["gaze"].last_raw_yaw
            st.write(f"**Raw model outputs:**")
            st.write(f"Pitch: {np.rad2deg(raw_pitch):.2f}° ({raw_pitch:.4f} rad)")
            st.write(f"Yaw: {np.rad2deg(raw_yaw):.2f}° ({raw_yaw:.4f} rad)")
            
            st.write(f"**After processing (centered & scaled):**")
            st.write(f"Pitch: {np.rad2deg(pred_pitch):.2f}° ({pred_pitch:.4f} rad)")
            st.write(f"Yaw: {np.rad2deg(pred_yaw):.2f}° ({pred_yaw:.4f} rad)")
            st.write(f"**Gaze Zone:** {gaze_zone}")
            st.write(f"**Note:** Raw values are re-centered (+8° to pitch) and scaled (5x) for visualization")
    
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
        is_near_intersection=is_near_intersection,
        pitch_deg=np.rad2deg(pred_pitch),
        yaw_deg=np.rad2deg(pred_yaw)
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
            gaze_image = draw_gaze_vectors(driver_image, pred_pitch, pred_yaw, gt_pitch, gt_yaw, 
                                          flip_horizontal=needs_horizontal_flip, 
                                          flip_both_axes=needs_both_axes_flip)
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
        # Group scenarios by actual driver ID (not just date) first, then by maneuver type
        drivers_data = defaultdict(lambda: defaultdict(list))
        driver_ids = set()
        driver_info = {}  # Store display info for each driver
        
        for scenario_name, sample_dir in brain4cars_scenarios.items():
            # Try both locations for _gt.json
            gt_path = os.path.join(sample_dir, "_gt.json")
            if not os.path.exists(gt_path):
                # Try root level
                gt_path = os.path.join(BRAIN4CARS_DIR, f"{scenario_name}_gt.json")
            
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                clip_id = gt_data.get('clip_id', scenario_name)
                driver_id, driver_display_name = get_actual_driver_id(clip_id)
                
                # Skip Driver 8
                if driver_id == "Driver_8":
                    continue
                
                maneuver_type = gt_data.get('maneuver', 'unknown')
                driver_ids.add(driver_id)
                
                # Store display info
                if driver_id not in driver_info:
                    driver_info[driver_id] = {
                        'display_name': driver_display_name,
                        'dates': set()
                    }
                driver_info[driver_id]['dates'].add(extract_driver_date(clip_id))
                
                drivers_data[driver_id][maneuver_type].append((scenario_name, sample_dir, gt_data))
        
        # Create display names for drivers with scenario counts
        driver_display_names = {}
        
        # Custom sort: move Driver_1A to the end
        def custom_driver_sort(driver_id):
            if driver_id == "Driver_1A":
                return ("zzz", driver_id)  # Sort to end
            return ("aaa", driver_id)  # Sort normally
        
        sorted_driver_ids = sorted(driver_ids, key=custom_driver_sort)
        
        for driver_id in sorted_driver_ids:
            total = sum(len(scenarios) for scenarios in drivers_data[driver_id].values())
            base_name = driver_info[driver_id]['display_name']
            driver_display_names[driver_id] = f"{base_name} - {total} scenarios"
        
        # Display driver selector
        if sorted_driver_ids:
            selected_driver_id = st.selectbox(
                "Select Driver",
                sorted_driver_ids,
                format_func=lambda x: driver_display_names.get(x, x),
                key="brain4cars_driver"
            )
            
            # Get maneuvers for selected driver
            driver_maneuvers = drivers_data[selected_driver_id]
            
            # Display maneuver selector
            maneuver_display = {
                'lchange': '⬅️ Left Lane Change',
                'rchange': '➡️ Right Lane Change',
                'lturn': '↰ Left Turn',
                'rturn': '↱ Right Turn',
                'end_action': '⬆️ Straight Driving'
            }
            
            maneuver_keys = sorted([m for m in driver_maneuvers.keys() if m in maneuver_display])
            if maneuver_keys:
                selected_maneuver = st.selectbox(
                    "Select Maneuver Type", 
                    maneuver_keys,
                    format_func=lambda x: f"{maneuver_display.get(x, x)} ({len(driver_maneuvers[x])} samples)",
                    key="brain4cars_maneuver"
                )
                
                # Collect all frames from all samples of this maneuver for this driver
                all_frames = []
                for scenario_name, sample_dir, gt_data in driver_maneuvers[selected_maneuver]:
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
                        f"Frame Sequence ({total_frames} frames across {len(driver_maneuvers[selected_maneuver])} samples)",
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
                    
                    # Show which driver, sample and frame we're viewing
                    dates = sorted(driver_info[selected_driver_id]['dates'])
                    date_str = ", ".join(dates) if len(dates) <= 2 else f"{dates[0]}..."
                    st.caption(f"👤 {driver_display_names[selected_driver_id].split(' -')[0]} | 📹 Sample: {selected_frame['scenario']} | Frame: {selected_frame['frame_idx']:02d}")
                    
                    process_and_display(driver_image, road_image, ground_truth, show_images, show_details)
                else:
                    st.error("No frame images found for this maneuver type.")
            else:
                st.warning(f"No valid maneuvers found for {driver_display_names.get(selected_driver_id, selected_driver_id)}.")
        else:
            st.warning("No valid drivers found in Brain4Cars data.")

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