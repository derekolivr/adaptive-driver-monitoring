#!/usr/bin/env python3
"""
Preprocess Brain4Cars dataset for testing in the Adaptive Driver Monitoring System.

This script extracts representative frames from Brain4Cars video clips and creates
a consistent format matching the DashGaze processed data structure.
"""

import os
import cv2
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import scipy.io  # Import SciPy for .mat file handling


def extract_middle_frame(video_path: str) -> Tuple[bool, any]:
    """
    Extract the middle frame from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (success, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return False, None
    
    # Seek to middle frame
    middle_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    return ret, frame


def extract_mat_data(mat_path: str, middle_frame_idx: int) -> Dict:
    """Extracts contextual data from a .mat file for a specific frame."""
    try:
        mat_data = scipy.io.loadmat(mat_path)
        
        # Lane info: [current_lane, total_lanes, is_intersection]
        lane_info = mat_data.get('laneInfo', [[0, 0, 0]])[0]
        
        # Frame-specific data (speed)
        frame_data = mat_data.get('frame_data')
        speed = 0
        if frame_data is not None and frame_data.shape[1] > middle_frame_idx:
            # Access the struct for the middle frame
            middle_frame_struct = frame_data[0, middle_frame_idx]
            # Speed is usually the first element in the struct's data
            if len(middle_frame_struct) > 0 and len(middle_frame_struct[0]) > 0:
                 speed_data = middle_frame_struct[0][0]
                 if 'speed' in speed_data.dtype.names:
                    speed = speed_data['speed'][0][0]


        return {
            "lane_info": {
                "current_lane": int(lane_info[0]),
                "total_lanes": int(lane_info[1]),
            },
            "is_near_intersection": bool(lane_info[2]),
            "speed_mph": float(speed)
        }
    except Exception as e:
        print(f"    ⚠️  Error reading .mat file {mat_path}: {e}")
        return {
            "lane_info": {"current_lane": 0, "total_lanes": 0},
            "is_near_intersection": False,
            "speed_mph": 0
        }


def get_clip_id_from_path(path: str) -> str:
    """Extract clip ID from path."""
    return Path(path).stem.replace('video_', '').replace('params_', '')


def process_maneuver_samples(
    face_camera_dir: str,
    road_camera_dir: str,
    maneuver: str,
    output_dir: str,
    samples_per_maneuver: int = 3
) -> List[Dict]:
    """
    Process a subset of samples for a given maneuver type.
    
    Args:
        face_camera_dir: Directory containing face camera videos
        road_camera_dir: Directory containing road camera videos
        maneuver: Maneuver type (e.g., 'lchange', 'rturn')
        output_dir: Output directory for processed frames
        samples_per_maneuver: Number of samples to extract per maneuver
        
    Returns:
        List of processed sample metadata
    """
    processed_samples = []
    
    # Get all face camera video directories for this maneuver
    face_dirs = sorted(glob.glob(os.path.join(face_camera_dir, maneuver, "*")))
    
    if not face_dirs:
        print(f"  ⚠️  No data found for maneuver: {maneuver}")
        return processed_samples
    
    # Sample evenly distributed clips
    step = max(1, len(face_dirs) // samples_per_maneuver)
    selected_dirs = face_dirs[::step][:samples_per_maneuver]
    
    print(f"  Processing {len(selected_dirs)} samples for {maneuver}...")
    
    for face_dir in selected_dirs:
        clip_name = os.path.basename(face_dir)
        
        # Find face video
        face_videos = glob.glob(os.path.join(face_dir, "video_*.avi"))
        if not face_videos:
            print(f"    ⚠️  No face video found for {clip_name}")
            continue
        
        face_video_path = face_videos[0]
        
        # Find corresponding road video
        road_video_path = os.path.join(road_camera_dir, maneuver, f"{clip_name}.avi")
        if not os.path.exists(road_video_path):
            print(f"    ⚠️  No road video found for {clip_name}")
            continue

        # Find corresponding .mat file (in face camera directory)
        mat_files = glob.glob(os.path.join(face_dir, "params_*.mat"))
        if not mat_files:
            print(f"    ⚠️  No .mat file found for {clip_name}")
            continue
        mat_path = mat_files[0]
        
        # Extract frames
        # To get speed at middle frame, we need the index
        cap = cv2.VideoCapture(face_video_path)
        middle_frame_idx = 0
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_idx = total_frames // 2
            cap.release()

        success_face, face_frame = extract_middle_frame(face_video_path)
        success_road, road_frame = extract_middle_frame(road_video_path)
        
        if not success_face or not success_road:
            print(f"    ⚠️  Failed to extract frames for {clip_name}")
            continue
        
        # Save frames
        output_prefix = f"{maneuver}_{clip_name}"
        driver_output_path = os.path.join(output_dir, f"{output_prefix}_driver.jpg")
        road_output_path = os.path.join(output_dir, f"{output_prefix}_road.jpg")
        
        cv2.imwrite(driver_output_path, face_frame)
        cv2.imwrite(road_output_path, road_frame)
        
        # Extract data from .mat file
        context_data = extract_mat_data(mat_path, middle_frame_idx)

        # Create metadata file
        metadata = {
            "source": "Brain4Cars",
            "maneuver": maneuver,
            "clip_id": clip_name,
            "face_video": face_video_path,
            "road_video": road_video_path,
            "note": "No gaze ground truth available for Brain4Cars dataset"
        }
        metadata.update(context_data)
        
        metadata_path = os.path.join(output_dir, f"{output_prefix}_gt.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        processed_samples.append(metadata)
        print(f"    ✓ Processed {clip_name}")
    
    return processed_samples


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Brain4Cars dataset for driver monitoring system"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Brain4Cars_data/brain4cars_data",
        help="Path to Brain4Cars data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/brain4cars_processed",
        help="Output directory for processed samples"
    )
    parser.add_argument(
        "--samples-per-maneuver",
        type=int,
        default=3,
        help="Number of samples to extract per maneuver type"
    )
    parser.add_argument(
        "--maneuvers",
        nargs='+',
        default=['lchange', 'rchange', 'lturn', 'rturn', 'end_action'],
        help="Maneuver types to process"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    face_camera_dir = os.path.join(args.data_dir, "face_camera")
    road_camera_dir = os.path.join(args.data_dir, "road_camera")
    output_dir = args.output_dir
    
    # Verify input directories exist
    if not os.path.exists(face_camera_dir):
        print(f"❌ Error: Face camera directory not found: {face_camera_dir}")
        return
    
    if not os.path.exists(road_camera_dir):
        print(f"❌ Error: Road camera directory not found: {road_camera_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚗 Brain4Cars Dataset Preprocessing")
    print(f"   Input: {args.data_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Samples per maneuver: {args.samples_per_maneuver}")
    print()
    
    # Process each maneuver type
    all_samples = []
    for maneuver in args.maneuvers:
        print(f"📁 Processing maneuver: {maneuver}")
        samples = process_maneuver_samples(
            face_camera_dir,
            road_camera_dir,
            maneuver,
            output_dir,
            args.samples_per_maneuver
        )
        all_samples.extend(samples)
        print()
    
    # Summary
    print(f"✅ Preprocessing complete!")
    print(f"   Total samples processed: {len(all_samples)}")
    print(f"   Output directory: {output_dir}")
    print()
    print("📊 Summary by maneuver:")
    maneuver_counts = {}
    for sample in all_samples:
        maneuver = sample['maneuver']
        maneuver_counts[maneuver] = maneuver_counts.get(maneuver, 0) + 1
    
    for maneuver, count in sorted(maneuver_counts.items()):
        print(f"   {maneuver}: {count} samples")


if __name__ == "__main__":
    main()

