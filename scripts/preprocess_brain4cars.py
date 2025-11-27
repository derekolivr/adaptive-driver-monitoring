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
import scipy.io
import numpy as np


def extract_frame_sequence(video_path: str, num_frames: int = 10, frame_gap: int = 2, frame_start: int = None, frame_end: int = None) -> Tuple[List[any], List[int]]:
    """
    Extracts a sequence of frames from a video with gaps for smoother playback.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_gap: Extract every Nth frame (e.g., 3 = every 3rd frame)
        frame_start: Optional start frame index (for maneuver window)
        frame_end: Optional end frame index (for maneuver window)
        
    Returns:
        Tuple of (list of frames, list of frame indices)
    """
    frames = []
    frame_indices = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames, frame_indices
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If maneuver window is specified, extract from that range
    if frame_start is not None and frame_end is not None:
        maneuver_frames = frame_end - frame_start + 1
        frames_needed = num_frames * frame_gap
        
        # Adjust parameters if maneuver window is too small
        if maneuver_frames < frames_needed:
            # Try reducing frame_gap first
            frame_gap = max(1, maneuver_frames // num_frames)
            frames_needed = num_frames * frame_gap
            
            # If still too small, reduce num_frames
            if maneuver_frames < frames_needed:
                num_frames = maneuver_frames // frame_gap
                if num_frames < 5:  # Minimum 5 frames
                    frame_gap = 1
                    num_frames = maneuver_frames
                frames_needed = num_frames * frame_gap
        
        # Center the extraction within the maneuver window
        start_frame = frame_start + (maneuver_frames - frames_needed) // 2
    else:
        # Default behavior: extract from middle of video
        frames_needed = num_frames * frame_gap
        if total_frames < frames_needed:
            cap.release()
            return frames, frame_indices
        start_frame = (total_frames - frames_needed) // 2
    
    for i in range(num_frames):
        frame_idx = start_frame + (i * frame_gap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_indices.append(frame_idx)
        else:
            break
            
    cap.release()
    return frames, frame_indices


def extract_mat_data(mat_path: str, frame_indices: List[int]) -> Dict:
    """
    Extracts contextual data from a .mat file.
    
    Args:
        mat_path: Path to .mat file
        frame_indices: List of video frame indices to extract speed for
    
    Returns:
        Dictionary with lane info, intersection status, and speed data
    """
    try:
        mat_data = scipy.io.loadmat(mat_path)
        
        # Extract params struct
        params = mat_data['params'][0, 0]
        
        # Parse lane info (comma-separated string: "current_lane,total_lanes,is_intersection")
        lane_info_str = str(params['laneInfo'][0])
        lane_parts = [int(x) for x in lane_info_str.split(',')]
        
        # Extract frame data
        frame_data = params['frame_data'][0]
        frame_start = int(params['frame_start'][0, 0])
        frame_end = int(params['frame_end'][0, 0])
        
        # Extract speeds for each frame index
        speeds = []
        for frame_idx in frame_indices:
            # Only extract speed if frame is within the maneuver window
            if frame_start <= frame_idx + 1 <= frame_end and frame_idx < len(frame_data):
                frame = frame_data[frame_idx]
                if hasattr(frame, 'dtype') and frame.dtype.names and 'speed' in frame.dtype.names:
                    speed_obj = frame['speed'][0, 0]
                    if hasattr(speed_obj, 'shape') and speed_obj.size > 0:
                        speed_val = float(speed_obj[0, 0])
                        speeds.append(speed_val)
                    else:
                        speeds.append(0.0)
                else:
                    speeds.append(0.0)
            else:
                speeds.append(0.0)
        
        # Use average speed across frames
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        return {
            "lane_info": {
                "current_lane": int(lane_parts[0]) if len(lane_parts) > 0 else 0,
                "total_lanes": int(lane_parts[1]) if len(lane_parts) > 1 else 0,
            },
            "is_near_intersection": bool(lane_parts[2]) if len(lane_parts) > 2 else False,
            "speed_mph": float(avg_speed),
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
        }
    except Exception as e:
        print(f"    ⚠️  Error reading .mat file {mat_path}: {e}")
        return {
            "lane_info": {"current_lane": 0, "total_lanes": 0},
            "is_near_intersection": False,
            "speed_mph": 0.0,
            "frame_start": 0,
            "frame_end": 0,
        }


def get_clip_id_from_path(path: str) -> str:
    """Extract clip ID from path."""
    return Path(path).stem.replace('video_', '').replace('params_', '')


def process_maneuver_samples(
    face_camera_dir: str,
    road_camera_dir: str,
    maneuver: str,
    output_dir: str,
    samples_per_maneuver: int = 5,
    frames_per_sample: int = 10,
    frame_gap: int = 2,
    driver_date: str = None
) -> List[Dict]:
    """
    Process a subset of samples for a given maneuver type.
    
    Args:
        face_camera_dir: Directory containing face camera videos
        road_camera_dir: Directory containing road camera videos
        maneuver: Maneuver type (e.g., 'lchange', 'rturn')
        output_dir: Output directory for processed frames
        samples_per_maneuver: Number of samples to extract per maneuver
        frames_per_sample: Number of frames to extract from each video
        frame_gap: Gap between extracted frames (every Nth frame)
        driver_date: Filter by date prefix (e.g., '20141019') for consistent driver
        
    Returns:
        List of processed sample metadata
    """
    processed_samples = []
    
    # Get all face camera video directories for this maneuver
    face_dirs = sorted(glob.glob(os.path.join(face_camera_dir, maneuver, "*")))
    
    # Filter by driver date if specified
    if driver_date:
        face_dirs = [d for d in face_dirs if os.path.basename(d).startswith(driver_date)]
        print(f"  Filtered to {len(face_dirs)} clips from driver date {driver_date}")
    
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
        
        # First, get the maneuver window from .mat file
        try:
            mat_data = scipy.io.loadmat(mat_path)
            params = mat_data['params'][0, 0]
            frame_start = int(params['frame_start'][0, 0])
            frame_end = int(params['frame_end'][0, 0])
        except Exception as e:
            print(f"    ⚠️  Error reading maneuver window from .mat file: {e}")
            continue
        
        # Extract frame sequences with gaps from within the maneuver window
        face_frames, face_indices = extract_frame_sequence(face_video_path, frames_per_sample, frame_gap, frame_start, frame_end)
        road_frames, road_indices = extract_frame_sequence(road_video_path, frames_per_sample, frame_gap, frame_start, frame_end)
        
        if not face_frames or not road_frames or len(face_frames) != len(road_frames):
            print(f"    ⚠️  Failed to extract sufficient frames for {clip_name}")
            continue
        
        # Create a subdirectory for this sample's frames
        sample_output_dir = os.path.join(output_dir, f"{maneuver}_{clip_name}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Save frames
        for i, (face_frame, road_frame) in enumerate(zip(face_frames, road_frames)):
            cv2.imwrite(os.path.join(sample_output_dir, f"driver_{i:02d}.jpg"), face_frame)
            cv2.imwrite(os.path.join(sample_output_dir, f"road_{i:02d}.jpg"), road_frame)
        
        # Extract data from .mat file using actual frame indices
        context_data = extract_mat_data(mat_path, face_indices)
        
        # Create metadata file
        metadata = {
            "source": "Brain4Cars",
            "maneuver": maneuver,
            "clip_id": clip_name,
            "num_frames": len(face_frames),
            "frame_gap": frame_gap,
            "driver_date": clip_name[:8],  # Extract date from clip name
            "note": "No gaze ground truth available for Brain4Cars dataset"
        }
        metadata.update(context_data)
        
        metadata_path = os.path.join(sample_output_dir, "_gt.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        processed_samples.append(metadata)
        print(f"    ✓ Processed {clip_name} (Speed: {context_data['speed_mph']:.1f} MPH)")
    
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
        default=5,
        help="Number of samples to extract per maneuver type"
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=10,
        help="Number of frames to extract from each video to create a sequence"
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=2,
        help="Gap between extracted frames (e.g., 2 = every 2nd frame for smoother playback)"
    )
    parser.add_argument(
        "--driver-date",
        type=str,
        default="20141019",
        help="Filter clips by date prefix for consistent driver (e.g., '20141019')"
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
    print(f"   Driver date filter: {args.driver_date}")
    print(f"   Samples per maneuver: {args.samples_per_maneuver}")
    print(f"   Frames per sample: {args.frames_per_sample}")
    print(f"   Frame gap: {args.frame_gap}")
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
            args.samples_per_maneuver,
            args.frames_per_sample,
            args.frame_gap,
            args.driver_date
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
    
    # Print speed statistics
    speeds = [s['speed_mph'] for s in all_samples if s.get('speed_mph', 0) > 0]
    if speeds:
        print(f"\n🚗 Speed statistics:")
        print(f"   Average: {np.mean(speeds):.1f} MPH")
        print(f"   Min: {np.min(speeds):.1f} MPH")
        print(f"   Max: {np.max(speeds):.1f} MPH")


if __name__ == "__main__":
    main()
