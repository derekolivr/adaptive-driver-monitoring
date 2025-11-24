"""
Simple linear calibration for gaze predictions.

Instead of retraining the model, this script finds a simple linear transformation:
    predicted_calibrated = scale * predicted_raw + offset

This is much more robust and prevents overfitting.
"""

import numpy as np
import json
import glob
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm

# Add parent and src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gaze_tracker import GazeTracker


def collect_predictions_and_ground_truth(model_path, data_dir):
    """
    Run the model on all test data and collect predictions vs ground truth.
    """
    print(f"Loading model from {model_path}...")
    tracker = GazeTracker(model_path=model_path)
    
    print(f"Collecting predictions from {data_dir}...")
    predictions = []
    ground_truths = []
    
    gt_files = sorted(glob.glob(os.path.join(data_dir, "*_gt.json")))
    
    for gt_path in tqdm(gt_files, desc="Processing frames"):
        base_name = os.path.basename(gt_path).replace("_gt.json", "")
        driver_path = os.path.join(data_dir, f"{base_name}_driver.jpg")
        
        if os.path.exists(driver_path):
            # Load ground truth
            with open(gt_path, 'r') as f:
                gt = json.load(f)
            
            # Get model prediction
            image = Image.open(driver_path)
            pred_pitch, pred_yaw = tracker.predict_gaze(image)
            
            # Store in degrees for easier interpretation
            predictions.append([np.rad2deg(pred_pitch), np.rad2deg(pred_yaw)])
            ground_truths.append([gt['elevation_deg'], gt['azimuth_deg']])
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    print(f"\nCollected {len(predictions)} samples.")
    return predictions, ground_truths


def compute_linear_calibration(predictions, ground_truths):
    """
    Compute optimal linear transformation: gt = scale * pred + offset
    Uses least squares regression.
    """
    print("\nComputing linear calibration...")
    
    # Separate pitch and yaw
    pred_pitch = predictions[:, 0].reshape(-1, 1)
    pred_yaw = predictions[:, 1].reshape(-1, 1)
    
    gt_pitch = ground_truths[:, 0]
    gt_yaw = ground_truths[:, 1]
    
    # Fit linear regression: gt = scale * pred + offset
    # Using numpy's least squares
    
    # For pitch: add bias term
    X_pitch = np.hstack([pred_pitch, np.ones((len(pred_pitch), 1))])
    pitch_params = np.linalg.lstsq(X_pitch, gt_pitch, rcond=None)[0]
    pitch_scale, pitch_offset = pitch_params
    
    # For yaw: add bias term
    X_yaw = np.hstack([pred_yaw, np.ones((len(pred_yaw), 1))])
    yaw_params = np.linalg.lstsq(X_yaw, gt_yaw, rcond=None)[0]
    yaw_scale, yaw_offset = yaw_params
    
    # Compute calibrated predictions
    calibrated_pitch = pitch_scale * pred_pitch.flatten() + pitch_offset
    calibrated_yaw = yaw_scale * pred_yaw.flatten() + yaw_offset
    
    # Compute errors before and after calibration
    error_pitch_before = np.mean(np.abs(pred_pitch.flatten() - gt_pitch))
    error_yaw_before = np.mean(np.abs(pred_yaw.flatten() - gt_yaw))
    
    error_pitch_after = np.mean(np.abs(calibrated_pitch - gt_pitch))
    error_yaw_after = np.mean(np.abs(calibrated_yaw - gt_yaw))
    
    print("\n" + "="*70)
    print("Calibration Results")
    print("="*70)
    print(f"\nPitch Calibration:")
    print(f"  Formula: calibrated = {pitch_scale:.4f} * raw + {pitch_offset:.4f}")
    print(f"  Error before: {error_pitch_before:.2f}°")
    print(f"  Error after:  {error_pitch_after:.2f}°")
    print(f"  Improvement:  {error_pitch_before - error_pitch_after:.2f}°")
    
    print(f"\nYaw Calibration:")
    print(f"  Formula: calibrated = {yaw_scale:.4f} * raw + {yaw_offset:.4f}")
    print(f"  Error before: {error_yaw_before:.2f}°")
    print(f"  Error after:  {error_yaw_after:.2f}°")
    print(f"  Improvement:  {error_yaw_before - error_yaw_after:.2f}°")
    
    print("\n" + "="*70)
    
    calibration_params = {
        "pitch_scale": float(pitch_scale),
        "pitch_offset": float(pitch_offset),
        "yaw_scale": float(yaw_scale),
        "yaw_offset": float(yaw_offset),
        "error_pitch_before": float(error_pitch_before),
        "error_pitch_after": float(error_pitch_after),
        "error_yaw_before": float(error_yaw_before),
        "error_yaw_after": float(error_yaw_after),
    }
    
    return calibration_params


def save_calibration(params, output_path="gaze_calibration.json"):
    """Save calibration parameters to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"\nCalibration parameters saved to {output_path}")


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "gaze_tracker_endterm.pth"  # Use the original MpiiFaceGaze model
    DATA_DIR = os.path.join("test_data", "dashgaze_processed")
    OUTPUT_FILE = "gaze_calibration.json"
    
    print("="*70)
    print("Gaze Calibration Script")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_DIR}")
    print("="*70)
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Data directory '{DATA_DIR}' not found!")
        print("Please run 'python scripts/preprocess_dashgaze.py' first.")
        sys.exit(1)
    
    # Collect predictions
    predictions, ground_truths = collect_predictions_and_ground_truth(MODEL_PATH, DATA_DIR)
    
    # Compute calibration
    calibration_params = compute_linear_calibration(predictions, ground_truths)
    
    # Save calibration
    save_calibration(calibration_params, OUTPUT_FILE)
    
    print("\n✅ Calibration complete!")
    print(f"To use this calibration, update src/gaze_tracker.py to apply these transformations.")

