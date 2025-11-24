import numpy as np

class FusionEngine:
    def __init__(self):
        """
        Initializes the Fusion Engine.
        This engine combines gaze and road context to determine driver focus.
        """
        # Define gaze angle thresholds in DEGREES for different zones.
        # These values may need tuning based on camera position.
        self.gaze_zones = {
            "Road Ahead": {"yaw_range": (-15, 15), "pitch_range": (-10, 5)},
            "Left Mirror/Window": {"yaw_range": (-50, -20), "pitch_range": (-15, 10)},
            "Right Mirror/Window": {"yaw_range": (20, 50), "pitch_range": (-15, 10)},
            "Center Console/Radio": {"yaw_range": (-20, 20), "pitch_range": (10, 40)},
            "Rear-view Mirror": {"yaw_range": (-15, 15), "pitch_range": (-30, -15)},
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

    def assess_driver_state(self, gaze_zone, road_objects):
        """
        Provides a high-level assessment of the driver's state based on gaze and road context.
        """
        # Rule-based assessment using only gaze tracking and road context
        if gaze_zone == "Looking Elsewhere":
            return f"WARNING: Driver's gaze is significantly off-road!"
        
        if gaze_zone != "Road Ahead":
            # Alert for visual distraction
            return f"WARNING: Driver is looking at {gaze_zone} instead of the road."

        # Driver is looking at road ahead - check road context
        if "person" in road_objects:
            return "CAUTION: Pedestrian detected. Driver is appropriately focused on the road."
        
        if len(road_objects) > 0:
            unique_objects = ', '.join(set(road_objects))
            return f"Driver is focused on the road ahead. Detected: {unique_objects}."

        return "Driver is focused on the road ahead. No critical objects detected."
