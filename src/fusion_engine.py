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

    def assess_driver_state(self, gaze_zone, road_objects, maneuver=None, is_near_intersection=False, speed_mph=0):
        """
        Provides a high-level assessment of the driver's state based on gaze and road context.
        
        Args:
            gaze_zone (str): The classified gaze zone of the driver.
            road_objects (list): A list of detected objects on the road.
            maneuver (str, optional): The current driving maneuver (e.g., 'lchange').
            is_near_intersection (bool, optional): True if the vehicle is near an intersection.
            speed_mph (float, optional): Vehicle speed in miles per hour.
        """
        # --- Rule-based assessment using integrated context ---
        
        # Rule 1: CRITICAL - Looking down at phone/lap is always dangerous
        if gaze_zone in ["Down (Phone/Lap)", "Center Console/Radio"]:
            if "person" in road_objects:
                return f"⚠️ CRITICAL: Driver looking at {gaze_zone} with pedestrian present!"
            if speed_mph > 25:
                return f"⚠️ CRITICAL: Driver distracted by {gaze_zone} at {speed_mph:.0f} MPH!"
            if is_near_intersection:
                return f"⚠️ CRITICAL: Driver looking at {gaze_zone} near intersection!"
            return f"⚠️ WARNING: Driver distracted by {gaze_zone}!"
        
        # Rule 2: High-priority warning for completely off-road gaze
        if gaze_zone == "Looking Elsewhere":
            if speed_mph > 15:
                return f"⚠️ CRITICAL: Driver's gaze significantly off-road at {speed_mph:.0f} MPH!"
            return "⚠️ WARNING: Driver's gaze is significantly off-road!"

        # Rule 3: Context-dependent severe warnings
        is_high_risk_context = is_near_intersection or "person" in road_objects or len(road_objects) > 2
        if gaze_zone not in ["Road Ahead", "Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"] and is_high_risk_context:
            if "person" in road_objects:
                return f"⚠️ CRITICAL: Driver not watching pedestrian! Looking at {gaze_zone}!"
            if is_near_intersection:
                return f"⚠️ WARNING: Driver looking at {gaze_zone} near intersection!"
            return f"⚠️ WARNING: Driver looking at {gaze_zone} in complex traffic!"

        # Rule 4: Safe, expected behavior during maneuvers (ONLY for appropriate mirror checks)
        if maneuver == 'lchange' and gaze_zone == "Left Mirror/Window":
            return "✓ Driver safely checking left mirror for lane change."
        if maneuver == 'rchange' and gaze_zone == "Right Mirror/Window":
            return "✓ Driver safely checking right mirror for lane change."
        if maneuver in ['lturn', 'rturn'] and gaze_zone in ["Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
             return f"✓ Driver checking surroundings for turn."

        # Rule 5: Speed-dependent warnings for mirror checks (when NOT during expected maneuver)
        if gaze_zone in ["Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
            if speed_mph > 50 and not maneuver:
                return f"⚠️ CAUTION: Extended mirror check at high speed ({speed_mph:.0f} MPH)."
            return f"Driver checking mirrors (speed: {speed_mph:.0f} MPH)."

        # Rule 6: Driver is looking ahead - provide context summary
        if "person" in road_objects:
            return "✓ GOOD: Pedestrian detected. Driver is appropriately focused on the road."
        
        if len(road_objects) > 0:
            unique_objects = ', '.join(set(road_objects))
            return f"✓ Driver focused on road. Detected: {unique_objects}."

        if speed_mph > 0:
            return f"✓ Driver focused on road ahead ({speed_mph:.0f} MPH)."
        
        return "✓ Driver is focused on the road ahead."
