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
        
        # Rule 1: High-priority warning for off-road gaze near intersections or with pedestrians
        is_high_risk_context = is_near_intersection or "person" in road_objects
        if gaze_zone not in ["Road Ahead", "Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"] and is_high_risk_context:
            return f"WARNING: Driver looking at {gaze_zone} near an intersection or pedestrian!"

        # Rule 2: Safe, expected behavior during maneuvers (e.g., checking mirrors for lane change)
        if maneuver == 'lchange' and gaze_zone == "Left Mirror/Window":
            return "Driver is safely checking mirror for lane change."
        if maneuver == 'rchange' and gaze_zone == "Right Mirror/Window":
            return "Driver is safely checking mirror for lane change."
        if maneuver in ['lturn', 'rturn'] and gaze_zone in ["Left Mirror/Window", "Right Mirror/Window"]:
             return "Driver is checking surroundings for turn."

        # Rule 3: General off-road gaze warnings
        if gaze_zone not in ["Road Ahead"]:
            return f"CAUTION: Driver is looking at {gaze_zone} instead of the road."

        # Rule 4: Driver is looking ahead, provide context summary
        if "person" in road_objects:
            return "CAUTION: Pedestrian detected. Driver is appropriately focused on the road."
        
        if len(road_objects) > 0:
            unique_objects = ', '.join(set(road_objects))
            return f"Driver focused on road. Detected: {unique_objects}."

        return "Driver is focused on the road ahead. No critical objects detected."
