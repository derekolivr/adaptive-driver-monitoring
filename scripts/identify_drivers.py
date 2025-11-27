#!/usr/bin/env python3
"""
Analyze driver images to identify different people.
Since date alone isn't sufficient (Driver 1 has multiple people, Drivers 2-4 are the same),
we need to visually inspect or use image comparison.
"""

import os
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

def extract_date_from_clip_id(clip_id):
    """Extract date (YYYYMMDD) from clip_id."""
    match = re.match(r'(\d{8})_', clip_id)
    if match:
        return match.group(1)
    return None

def get_driver_image_path(scenario_dir):
    """Get the first driver image from a scenario directory."""
    # Try to find driver_00.jpg or any driver image
    for i in range(20):
        driver_path = os.path.join(scenario_dir, f"driver_{i:02d}.jpg")
        if os.path.exists(driver_path):
            return driver_path
    
    # Also check root level
    scenario_name = os.path.basename(os.path.normpath(scenario_dir))
    root_driver = os.path.join(os.path.dirname(scenario_dir), f"{scenario_name}_driver.jpg")
    if os.path.exists(root_driver):
        return root_driver
    
    return None

def analyze_driver_images():
    """Analyze driver images to identify different people."""
    brain4cars_dir = Path("test_data/brain4cars_processed")
    
    if not brain4cars_dir.exists():
        print(f"Error: {brain4cars_dir} does not exist")
        return
    
    # Group by date first
    scenarios_by_date = defaultdict(list)
    
    # Find all scenarios
    for scenario_dir in brain4cars_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        
        gt_path = scenario_dir / "_gt.json"
        if not gt_path.exists():
            # Try root level
            scenario_name = scenario_dir.name
            gt_path = brain4cars_dir / f"{scenario_name}_gt.json"
        
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                metadata = json.load(f)
            
            clip_id = metadata.get('clip_id', scenario_dir.name)
            date = extract_date_from_clip_id(clip_id)
            
            driver_image_path = get_driver_image_path(scenario_dir)
            if driver_image_path:
                scenarios_by_date[date].append({
                    'clip_id': clip_id,
                    'scenario': scenario_dir.name,
                    'driver_image': driver_image_path,
                    'maneuver': metadata.get('maneuver', 'unknown')
                })
    
    print("=" * 80)
    print("Driver Image Analysis")
    print("=" * 80)
    print("\nBased on user feedback:")
    print("- Driver 1 (20141019) has MULTIPLE different drivers")
    print("- Drivers 2, 3, 4 (20141025, 20141101, 20141102) are the SAME person")
    print("\n" + "=" * 80)
    
    # Show Driver 1 scenarios (need to split)
    print("\n📅 Date 20141019 (Driver 1 - needs splitting):")
    print(f"   Total scenarios: {len(scenarios_by_date.get('20141019', []))}")
    print("\n   Sample driver images to inspect:")
    driver1_scenarios = scenarios_by_date.get('20141019', [])
    for i, scenario in enumerate(driver1_scenarios[:10], 1):  # Show first 10
        print(f"   {i}. {scenario['scenario']} ({scenario['maneuver']})")
        print(f"      Image: {scenario['driver_image']}")
    
    # Show Drivers 2-4 (should be merged)
    print("\n📅 Dates 20141025, 20141101, 20141102 (Same person - should be merged):")
    for date in ['20141025', '20141101', '20141102']:
        scenarios = scenarios_by_date.get(date, [])
        if scenarios:
            print(f"\n   {date}: {len(scenarios)} scenarios")
            if scenarios:
                print(f"      Sample: {scenarios[0]['scenario']}")
                print(f"      Image: {scenarios[0]['driver_image']}")
    
    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    print("1. Manually inspect driver images from 20141019 to identify different people")
    print("2. Create a driver mapping file (driver_mapping.json) that groups scenarios")
    print("   by actual driver identity, not just date")
    print("3. Update the Streamlit app to use this mapping")
    
    return scenarios_by_date

if __name__ == "__main__":
    analyze_driver_images()

