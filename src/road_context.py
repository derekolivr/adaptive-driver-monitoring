from ultralytics import YOLO
from PIL import Image

class RoadContext:
    def __init__(self):
        """
        Initializes the YOLOv8 model for road context object detection.
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")  # Use the nano version for speed
        print("YOLOv8 model loaded.")

        # Classes we care about for driver context
        self.target_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic light",
            11: "stop sign",
            0: "person",
        }

    def detect_objects(self, road_image: Image.Image):
        """
        Detects relevant objects in a road-facing image.
        Returns the annotated image and a list of detected object names.
        """
        results = self.model(road_image, verbose=False)
        
        annotated_image = results[0].plot() # This is a numpy array (BGR)
        annotated_image_pil = Image.fromarray(annotated_image[:, :, ::-1]) # Convert BGR to RGB

        detected_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id in self.target_classes:
                detected_objects.append(self.target_classes[class_id])
                
        return annotated_image_pil, detected_objects
