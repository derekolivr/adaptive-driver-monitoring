import numpy as np

class FusionEngine:
    def __init__(self):
        """
        Initializes the Fusion Engine.
        This engine combines gaze and road context to determine driver focus.
        """
        # Define gaze angle thresholds in DEGREES for different zones.
        # Updated based on DashGaze dataset analysis and visual inspection
        # Note: These zones account for typical driving head poses
        self.gaze_zones = {
            "Road Ahead": {"yaw_range": (-8, 8), "pitch_range": (-8, 8)},
            "Left Mirror/Window": {"yaw_range": (-60, -10), "pitch_range": (-25, 20)},
            "Right Mirror/Window": {"yaw_range": (10, 60), "pitch_range": (-25, 20)},
            "Rear-view Mirror": {"yaw_range": (-8, 8), "pitch_range": (-40, -10)},
            "Center Console/Radio": {"yaw_range": (-12, 12), "pitch_range": (20, 45)},
            "Down (Phone/Lap)": {"yaw_range": (-12, 12), "pitch_range": (8, 20)},
        }

    def classify_gaze_zone(self, pitch, yaw):
        """
        Classifies gaze angles (in radians) into a predefined zone.
        """
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)

        for zone, ranges in self.gaze_zones.items():
            if ranges["yaw_range"][0] < yaw_deg < ranges["yaw_range"][1] and \
               ranges["pitch_range"][0] < pitch_deg < ranges["pitch_range"][1]:
                return zone
        
        return "Looking Elsewhere"

    def assess_driver_state(self, gaze_zone, road_objects, maneuver=None, is_near_intersection=False):
        """
        Provides a high-level assessment of the driver's state based on gaze and road context.
        
        Args:
            gaze_zone (str): The classified gaze zone of the driver.
            road_objects (list): A list of detected objects on the road.
            maneuver (str, optional): The current driving maneuver (e.g., 'lchange').
            is_near_intersection (bool, optional): True if the vehicle is near an intersection.
        """
        # --- Rule-based assessment using integrated context ---
        # Yellow (CAUTION) range is wider, Red (WARNING/CRITICAL) is smaller
        
        # Rule 1: All distractions are CAUTION (yellow), no red warnings
        if gaze_zone in ["Down (Phone/Lap)", "Center Console/Radio"]:
            # Intersections are CAUTION (yellow)
            if is_near_intersection:
                return f"⚠️ CAUTION: Driver looking at {gaze_zone} near intersection."
            # Pedestrians are CAUTION (yellow)
            if "person" in road_objects:
                return f"⚠️ CAUTION: Driver looking at {gaze_zone} with pedestrian present."
            # Normal distraction - just note it, not a warning
            return f"Driver checking {gaze_zone}."
        
        # Rule 2: Off-road gaze - all CAUTION (yellow)
        if gaze_zone == "Looking Elsewhere":
            # Intersections are CAUTION (yellow)
            if is_near_intersection:
                return "⚠️ CAUTION: Driver's gaze off-road near intersection."
            # Pedestrians are CAUTION (yellow)
            if "person" in road_objects:
                return "⚠️ CAUTION: Driver's gaze off-road with pedestrian present."
            # Normal off-road glance - always CAUTION
            return "⚠️ CAUTION: Driver's gaze is significantly off-road."

        # Rule 3: Context-dependent - all CAUTION (yellow), no red
        # Only trigger for clearly distracting zones, not mirrors or safe zones
        distracting_zones = ["Down (Phone/Lap)", "Center Console/Radio", "Looking Elsewhere"]
        if gaze_zone in distracting_zones:
            # This is already handled by Rules 1 and 2, so skip
            pass
        elif gaze_zone not in ["Road Ahead", "Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
            # Other zones - all CAUTION (yellow)
            if is_near_intersection:
                return f"⚠️ CAUTION: Driver looking at {gaze_zone} near intersection."
            if "person" in road_objects:
                return f"⚠️ CAUTION: Driver looking at {gaze_zone} with pedestrian present."
            # For other zones, just note it without warning
            return f"Driver looking at {gaze_zone}."

        # Rule 4: Safe, expected behavior during maneuvers
        if maneuver == 'lchange' and gaze_zone in ["Left Mirror/Window", "Rear-view Mirror"]:
            return "✓ Driver safely checking mirrors for lane change."
        if maneuver == 'rchange' and gaze_zone in ["Right Mirror/Window", "Rear-view Mirror"]:
            return "✓ Driver safely checking mirrors for lane change."
        if maneuver in ['lturn', 'rturn'] and gaze_zone in ["Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
             return f"✓ Driver checking surroundings for turn."

        # Rule 5: Mirror checks
        if gaze_zone in ["Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
            return "✓ Driver checking mirrors."

        # Rule 6: Driver is looking ahead - provide context summary
        if "person" in road_objects:
            return "✓ GOOD: Pedestrian detected. Driver is appropriately focused on the road."
        
        if len(road_objects) > 0:
            unique_objects = ', '.join(set(road_objects))
            return f"✓ Driver focused on road. Detected: {unique_objects}."

        return "✓ Driver is focused on the road ahead."
