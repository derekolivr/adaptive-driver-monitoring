import numpy as np

class FusionEngine:
    def __init__(self):
        """
        Initializes the Fusion Engine.
        This engine combines gaze and road context to determine driver focus.
        """
        # Define gaze angle thresholds in DEGREES for different zones.
        # Based on DashGaze ground truth analysis:
        # - Normal road ahead: elevation 5.5° to 10.4° (POSITIVE = looking down at road)
        # - COORDINATE SYSTEM: Positive pitch = DOWN (road/phone), Negative pitch = UP (mirrors/sky)
        # - Zone check order matters - more specific zones should be defined clearly
        self.gaze_zones = {
            "Rear-view Mirror": {"yaw_range": (-12, 12), "pitch_range": (-50, -2)},  # Looking UP (negative pitch)
            "Road Ahead": {"yaw_range": (-12, 12), "pitch_range": (-2, 12)},  # GT: 5-10° typical, expanded tolerance
            "Down (Phone/Lap)": {"yaw_range": (-15, 15), "pitch_range": (12, 60)},  # Looking significantly DOWN (> 12°)
            "Left Mirror/Window": {"yaw_range": (-60, -12), "pitch_range": (-40, 25)},  # Left side
            "Right Mirror/Window": {"yaw_range": (12, 60), "pitch_range": (-40, 25)},  # Right side
        }

    def classify_gaze_zone(self, pitch, yaw):
        """
        Classifies gaze angles (in radians) into a predefined zone.
        """
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)

        for zone, ranges in self.gaze_zones.items():
            if ranges["yaw_range"][0] <= yaw_deg <= ranges["yaw_range"][1] and \
               ranges["pitch_range"][0] <= pitch_deg <= ranges["pitch_range"][1]:
                return zone
        
        return "Looking Elsewhere"

    def assess_driver_state(self, gaze_zone, road_objects, maneuver=None, is_near_intersection=False, pitch_deg=None, yaw_deg=None):
        """
        Provides a high-level assessment of the driver's state based on gaze and road context.
        
        Args:
            gaze_zone (str): The classified gaze zone of the driver.
            road_objects (list): A list of detected objects on the road.
            maneuver (str, optional): The current driving maneuver (e.g., 'lchange', 'rturn', 'lturn').
            is_near_intersection (bool, optional): True if the vehicle is near an intersection.
            pitch_deg (float, optional): Pitch angle in degrees for fine-grained assessment.
            yaw_deg (float, optional): Yaw angle in degrees for fine-grained assessment.
        """
        # --- Maneuver-aware safety assessment ---
        
        # CRITICAL RULE: Looking down during any active maneuver is ALWAYS unsafe
        if gaze_zone == "Down (Phone/Lap)":
            if maneuver and maneuver != 'end_action':
                return f"⚠️ CAUTION: Driver distracted (looking down) during {maneuver.replace('_', ' ')}!"
            if is_near_intersection:
                return "⚠️ CAUTION: Driver distracted (looking down) near intersection!"
            if "person" in road_objects:
                return "⚠️ CAUTION: Driver distracted (looking down) with pedestrian present!"
            if any(obj in road_objects for obj in ["car", "truck", "bus", "motorcycle"]):
                return "⚠️ CAUTION: Driver distracted (looking down) with vehicles ahead!"
            return "⚠️ CAUTION: Driver distracted - looking down (phone/lap area)."
        
        # MANEUVER-SPECIFIC SAFETY CHECKS (using fine-grained yaw/pitch if available)
        if maneuver and yaw_deg is not None:
            # RIGHT TURN: Driver should be looking right (positive yaw)
            if maneuver == 'rturn':
                if yaw_deg < -10:  # Looking left during right turn
                    return "⚠️ CAUTION: Driver looking LEFT during RIGHT turn - wrong direction!"
                elif yaw_deg > 5:  # Looking right (good for right turn)
                    if gaze_zone in ["Right Mirror/Window", "Road Ahead"]:
                        return "✓ SAFE: Driver correctly checking right for turn."
                elif gaze_zone == "Down (Phone/Lap)":
                    return "⚠️ CAUTION: Driver distracted (looking down) during turn!"
            
            # LEFT TURN: Driver should be looking left (negative yaw)
            elif maneuver == 'lturn':
                if yaw_deg > 10:  # Looking right during left turn
                    return "⚠️ CAUTION: Driver looking RIGHT during LEFT turn - wrong direction!"
                elif yaw_deg < -5:  # Looking left (good for left turn)
                    if gaze_zone in ["Left Mirror/Window", "Road Ahead"]:
                        return "✓ SAFE: Driver correctly checking left for turn."
                elif gaze_zone == "Down (Phone/Lap)":
                    return "⚠️ CAUTION: Driver distracted (looking down) during turn!"
            
            # RIGHT LANE CHANGE: Should check right mirror/blind spot
            elif maneuver == 'rchange':
                if gaze_zone in ["Right Mirror/Window", "Rear-view Mirror"]:
                    return "✓ SAFE: Driver checking mirrors for right lane change."
                elif gaze_zone == "Road Ahead" and yaw_deg is not None and yaw_deg > 5:
                    # Looking right (even if not fully in mirror zone) is acceptable for right lane change
                    return "✓ SAFE: Driver looking right for right lane change."
                elif gaze_zone == "Left Mirror/Window":
                    return "⚠️ CAUTION: Driver checking wrong side for right lane change!"
                elif gaze_zone == "Road Ahead" and yaw_deg is not None and yaw_deg < -5:
                    return "⚠️ CAUTION: Driver looking left during right lane change - wrong direction!"
                elif gaze_zone == "Down (Phone/Lap)":
                    return "⚠️ CAUTION: Driver distracted (looking down) during lane change!"
            
            # LEFT LANE CHANGE: Should check left mirror/blind spot
            elif maneuver == 'lchange':
                if gaze_zone in ["Left Mirror/Window", "Rear-view Mirror"]:
                    return "✓ SAFE: Driver checking mirrors for left lane change."
                elif gaze_zone == "Road Ahead" and yaw_deg is not None and yaw_deg < -5:
                    # Looking left (even if not fully in mirror zone) is acceptable for left lane change
                    return "✓ SAFE: Driver looking left for left lane change."
                elif gaze_zone == "Right Mirror/Window":
                    return "⚠️ CAUTION: Driver checking wrong side for left lane change!"
                elif gaze_zone == "Road Ahead" and yaw_deg is not None and yaw_deg > 5:
                    return "⚠️ CAUTION: Driver looking right during left lane change - wrong direction!"
                elif gaze_zone == "Down (Phone/Lap)":
                    return "⚠️ CAUTION: Driver distracted (looking down) during lane change!"
        
        
        # Off-road gaze warnings
        if gaze_zone == "Looking Elsewhere":
            if maneuver and maneuver != 'end_action':
                return f"⚠️ CAUTION: Driver's gaze off-road during {maneuver.replace('_', ' ')}!"
            if is_near_intersection:
                return "⚠️ CAUTION: Driver's gaze off-road near intersection."
            if "person" in road_objects:
                return "⚠️ CAUTION: Driver's gaze off-road with pedestrian present."
            return "⚠️ CAUTION: Driver's gaze is significantly off-road."

        # Context-dependent warnings for unusual gaze zones
        if gaze_zone not in ["Road Ahead", "Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror", 
                             "Down (Phone/Lap)", "Looking Elsewhere"]:
            if is_near_intersection or "person" in road_objects:
                return f"⚠️ CAUTION: Driver looking at {gaze_zone} in critical situation."
            return f"Driver looking at {gaze_zone}."

        # Mirror checks during straight driving (no active maneuver)
        if not maneuver or maneuver == 'end_action':
            if gaze_zone in ["Left Mirror/Window", "Right Mirror/Window", "Rear-view Mirror"]:
                return "✓ Driver checking mirrors - good situational awareness."

        # Driver looking ahead - provide context
        if gaze_zone == "Road Ahead":
            if "person" in road_objects:
                return "✓ SAFE: Pedestrian detected. Driver focused on road."
            if len(road_objects) > 0:
                unique_objects = ', '.join(set(road_objects))
                return f"✓ SAFE: Driver focused on road. Detected: {unique_objects}."
            if maneuver and maneuver != 'end_action':
                return f"✓ Driver monitoring road during {maneuver.replace('_', ' ')}."
            return "✓ SAFE: Driver focused on the road ahead."

        # Default safe state
        return "✓ Driver maintaining awareness."
