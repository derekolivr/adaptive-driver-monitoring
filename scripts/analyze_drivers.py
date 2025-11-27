#!/usr/bin/env python3
"""
Analyze Brain4Cars test data to identify different drivers based on dates.
The date prefix in clip_id (e.g., 20141019) likely corresponds to different drivers.
"""

import os
import json
import re
from collections import defaultdict
from pathlib import Path

def extract_date_from_clip_id(clip_id):
    """Extract date (YYYYMMDD) from clip_id like '20141019_091035_1106_1256'"""
    match = re.match(r'(\d{8})_', clip_id)
    if match:
        return match.group(1)
    return None

def analyze_drivers():
    """Analyze all Brain4Cars scenarios and group by date (driver)."""
    brain4cars_dir = Path("test_data/brain4cars_processed")
    
    if not brain4cars_dir.exists():
        print(f"Error: {brain4cars_dir} does not exist")
        return
    
    # Group scenarios by date
    scenarios_by_date = defaultdict(lambda: {
        'maneuvers': defaultdict(list),
        'total_scenarios': 0
    })
    
    # Find all _gt.json files (both formats)
    for gt_file in brain4cars_dir.rglob("*_gt.json"):
        try:
            with open(gt_file, 'r') as f:
                metadata = json.load(f)
            
            clip_id = metadata.get('clip_id', '')
            maneuver = metadata.get('maneuver', 'unknown')
            date = extract_date_from_clip_id(clip_id)
            
            if date:
                scenarios_by_date[date]['maneuvers'][maneuver].append({
                    'clip_id': clip_id,
                    'path': str(gt_file.parent),
                    'metadata': metadata
                })
                scenarios_by_date[date]['total_scenarios'] += 1
        except Exception as e:
            print(f"Error reading {gt_file}: {e}")
    
    # Print analysis
    print("=" * 80)
    print("Brain4Cars Driver Analysis (Grouped by Date)")
    print("=" * 80)
    print(f"\nTotal unique dates (likely drivers): {len(scenarios_by_date)}\n")
    
    # Sort dates chronologically
    sorted_dates = sorted(scenarios_by_date.keys())
    
    for date in sorted_dates:
        info = scenarios_by_date[date]
        print(f"\nDate: {date} (Driver {sorted_dates.index(date) + 1})")
        print(f"  Total scenarios: {info['total_scenarios']}")
        print(f"  Maneuvers:")
        for maneuver, scenarios in sorted(info['maneuvers'].items()):
            print(f"    - {maneuver}: {len(scenarios)} scenarios")
    
    print("\n" + "=" * 80)
    print("Summary by Maneuver Type:")
    print("=" * 80)
    
    # Count by maneuver across all drivers
    maneuver_counts = defaultdict(lambda: defaultdict(int))
    for date, info in scenarios_by_date.items():
        for maneuver in info['maneuvers'].keys():
            maneuver_counts[maneuver][date] = len(info['maneuvers'][maneuver])
    
    for maneuver in sorted(maneuver_counts.keys()):
        print(f"\n{maneuver}:")
        for date in sorted(maneuver_counts[maneuver].keys()):
            count = maneuver_counts[maneuver][date]
            driver_num = sorted_dates.index(date) + 1
            print(f"  Driver {driver_num} ({date}): {count} scenarios")
    
    return scenarios_by_date, sorted_dates

if __name__ == "__main__":
    analyze_drivers()

