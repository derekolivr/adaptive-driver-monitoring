#!/usr/bin/env python3
"""
Verify that gaze zones are correctly mapped to the DashGaze ground truth coordinate system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion_engine import FusionEngine
import numpy as np

def test_gaze_zones():
    """Test gaze zone classification with corrected coordinate system."""
    fusion = FusionEngine()
    
    print("=" * 70)
    print("Gaze Zone Verification")
    print("=" * 70)
    print("\nCoordinate System:")
    print("- Positive pitch = looking DOWN (toward road/phone)")
    print("- Negative pitch = looking UP (toward mirrors/sky)")
    print("- Positive yaw = looking RIGHT")
    print("- Negative yaw = looking LEFT")
    print("\n" + "=" * 70)
    
    # Test cases based on DashGaze ground truth observations
    test_cases = [
        # (pitch_deg, yaw_deg, expected_zone, description)
        (7.0, 0.0, "Road Ahead", "Typical road ahead (GT: 5-10°)"),
        (10.0, -5.0, "Road Ahead", "Road ahead with slight left look"),
        (15.0, 0.0, "Down (Phone/Lap)", "Looking down at lap/phone"),
        (20.0, 2.0, "Down (Phone/Lap)", "Looking significantly down"),
        (-10.0, 0.0, "Rear-view Mirror", "Looking up at rearview mirror"),
        (-25.0, 3.0, "Rear-view Mirror", "Looking up"),
        (5.0, 25.0, "Right Mirror/Window", "Looking right"),
        (5.0, -30.0, "Left Mirror/Window", "Looking left"),
        (0.0, 0.0, "Road Ahead", "Straight ahead (edge case)"),
        (13.0, 0.0, "Down (Phone/Lap)", "Boundary: start of down zone"),
    ]
    
    print("\nTest Cases:")
    print("-" * 70)
    
    for pitch_deg, yaw_deg, expected, description in test_cases:
        pitch_rad = np.deg2rad(pitch_deg)
        yaw_rad = np.deg2rad(yaw_deg)
        actual = fusion.classify_gaze_zone(pitch_rad, yaw_rad)
        
        status = "✓" if actual == expected else "✗"
        color = "\033[92m" if actual == expected else "\033[91m"  # Green or Red
        reset = "\033[0m"
        
        print(f"{status} {color}Pitch: {pitch_deg:6.1f}°  Yaw: {yaw_deg:6.1f}° → {actual:20s}{reset}")
        print(f"   Description: {description}")
        if actual != expected:
            print(f"   Expected: {expected}")
        print()
    
    print("=" * 70)
    print("\nDashGaze Ground Truth Reference:")
    print("- Normal driving: elevation 5.5° to 10.4° (all positive)")
    print("- Azimuth range: -13.5° to 10.5° (left to right)")
    print("\nZone Definitions:")
    for zone, ranges in fusion.gaze_zones.items():
        print(f"  {zone:25s}: Yaw {ranges['yaw_range']}, Pitch {ranges['pitch_range']}")
    print("=" * 70)

if __name__ == "__main__":
    test_gaze_zones()

