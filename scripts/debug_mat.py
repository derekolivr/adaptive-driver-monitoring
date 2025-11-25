#!/usr/bin/env python3
"""
Debug script to understand the structure of Brain4Cars .mat files.
This will help us correctly extract speed and other contextual data.
"""

import scipy.io
import numpy as np
import glob
import os

def print_mat_structure(mat_data, indent=0):
    """Recursively print the structure of a .mat file."""
    prefix = "  " * indent
    
    if isinstance(mat_data, dict):
        for key, value in mat_data.items():
            if key.startswith('__'):
                continue
            print(f"{prefix}{key}: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"{prefix}  Shape: {value.shape}, Dtype: {value.dtype}")
                if value.dtype.names:
                    print(f"{prefix}  Struct fields: {value.dtype.names}")
                if value.size < 10 and value.dtype != object:
                    print(f"{prefix}  Values: {value}")
            
            if isinstance(value, (dict, np.ndarray)) and value.dtype == object:
                if isinstance(value, np.ndarray) and value.size > 0:
                    print(f"{prefix}  Inspecting first element:")
                    print_mat_structure(value.flat[0], indent + 2)

def extract_speed_test(mat_path):
    """Test different methods to extract speed from .mat file."""
    print("\n" + "="*80)
    print(f"Testing: {os.path.basename(mat_path)}")
    print("="*80)
    
    # Load with different options
    print("\n--- Loading with default options ---")
    mat_data = scipy.io.loadmat(mat_path)
    print_mat_structure(mat_data)
    
    print("\n--- Attempting to extract speed ---")
    
    # Method 1: Direct access to frame_data
    if 'frame_data' in mat_data:
        frame_data = mat_data['frame_data']
        print(f"\nframe_data found!")
        print(f"  Type: {type(frame_data)}")
        print(f"  Shape: {frame_data.shape}")
        print(f"  Dtype: {frame_data.dtype}")
        
        if frame_data.dtype.names:
            print(f"  Fields: {frame_data.dtype.names}")
            
            # Try to access the first frame
            if frame_data.shape[1] > 0:
                first_frame = frame_data[0, 0]
                print(f"\n  First frame type: {type(first_frame)}")
                print(f"  First frame dtype: {first_frame.dtype if hasattr(first_frame, 'dtype') else 'N/A'}")
                
                if hasattr(first_frame, 'dtype') and first_frame.dtype.names:
                    print(f"  First frame fields: {first_frame.dtype.names}")
                    
                    if 'speed' in first_frame.dtype.names:
                        speed = first_frame['speed']
                        print(f"\n  ✓ SUCCESS! Speed extracted: {speed}")
                        print(f"    Speed type: {type(speed)}")
                        print(f"    Speed shape: {speed.shape if hasattr(speed, 'shape') else 'N/A'}")
                        if hasattr(speed, 'shape') and len(speed.shape) > 0:
                            print(f"    Speed value: {speed[0, 0] if speed.size > 0 else 'empty'}")
    
    # Check frame_start and frame_end
    if 'frame_start' in mat_data:
        print(f"\nframe_start: {mat_data['frame_start']}")
    if 'frame_end' in mat_data:
        print(f"frame_end: {mat_data['frame_end']}")
    
    # Check laneInfo
    if 'laneInfo' in mat_data:
        lane_info = mat_data['laneInfo']
        print(f"\nlaneInfo: {lane_info}")
        print(f"  Shape: {lane_info.shape}")
        print(f"  Values: {lane_info[0] if lane_info.size > 0 else 'empty'}")

def main():
    """Main function to debug .mat files."""
    # Find a sample .mat file from each maneuver type
    base_dir = "data/Brain4Cars_data/brain4cars_data/face_camera"
    
    maneuvers = ['lchange', 'rchange', 'lturn', 'rturn', 'end_action']
    
    for maneuver in maneuvers:
        maneuver_dir = os.path.join(base_dir, maneuver)
        if not os.path.exists(maneuver_dir):
            continue
        
        # Find first .mat file
        mat_files = glob.glob(os.path.join(maneuver_dir, "*/params_*.mat"))
        if mat_files:
            extract_speed_test(mat_files[0])
            break  # Just test one file for now
    
    print("\n" + "="*80)
    print("Debug complete!")
    print("="*80)

if __name__ == "__main__":
    main()

