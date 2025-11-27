#!/usr/bin/env python3
"""Test the driver mapping logic."""

import re

def extract_driver_date(clip_id: str) -> str:
    """Extract date (YYYYMMDD) from clip_id."""
    match = re.match(r'(\d{8})_', clip_id)
    if match:
        return match.group(1)
    return "unknown"

def extract_time_from_clip_id(clip_id: str) -> int:
    """Extract time (HHMMSS) from clip_id."""
    match = re.match(r'\d{8}_(\d{6})_', clip_id)
    if match:
        return int(match.group(1))
    return 0

def get_actual_driver_id(clip_id: str) -> str:
    """Map clip_id to actual driver ID."""
    date = extract_driver_date(clip_id)
    
    # Merge dates 20141025, 20141101, 20141102 into one driver
    if date in ["20141025", "20141101", "20141102"]:
        return "Driver_2_20141025_20141101_20141102"
    
    # Split 20141019 by time of day
    if date == "20141019":
        time = extract_time_from_clip_id(clip_id)
        hour = time // 10000
        if hour < 12:
            return "Driver_1A_20141019_Morning"
        else:
            return "Driver_1B_20141019_Afternoon"
    
    if date != "unknown":
        return f"Driver_{date}"
    
    return "Driver_Unknown"

# Test cases
test_clips = [
    "20141019_085226_294_444",  # Morning (08:52)
    "20141019_091035_1106_1256",  # Morning (09:10)
    "20141019_132535_1229_1379",  # Afternoon (13:25)
    "20141019_141730_1106_1256",  # Afternoon (14:17)
    "20141025_141422_1062_1212",  # Should be Driver 2
    "20141101_153705_1_146",  # Should be Driver 2
    "20141102_103010_113_263",  # Should be Driver 2
    "20141105_135226_662_812",  # Should be separate driver
    "20141220_131116_579_729",  # Should be separate driver
]

print("Testing Driver Mapping Logic:")
print("=" * 60)
for clip_id in test_clips:
    driver_id = get_actual_driver_id(clip_id)
    date = extract_driver_date(clip_id)
    time = extract_time_from_clip_id(clip_id)
    hour = time // 10000 if time > 0 else 0
    print(f"{clip_id}")
    print(f"  Date: {date}, Time: {time:06d} ({hour:02d}:{(time%10000)//100:02d})")
    print(f"  Driver ID: {driver_id}")
    print()

