import torch
from torchvision import transforms
from PIL import Image
from gaze_model import get_gaze_model
import os
import numpy as np

class GazeTracker:
    def __init__(self, model_path="gaze_tracker_endterm.pth"):
        """
        Initializes the GazeTracker model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gaze model file not found at {model_path}")

        # Set device (MPS for Apple Silicon, CUDA for Nvidia, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"GazeTracker using device: {self.device}")

        self.model = get_gaze_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict_gaze(self, face_image: Image.Image, apply_calibration=True):
        """
        Predicts the gaze angles (pitch and yaw) from a PIL Image of a cropped face.
        
        Args:
            face_image: PIL Image of a cropped face
            apply_calibration: If True, applies DashGaze-specific calibration.
                             Set to False for other datasets like Brain4Cars.
        
        Returns:
            Tuple of (pitch, yaw) in radians
        """
        face_image = face_image.convert("RGB")
        image_tensor = self.transform(face_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
        
        raw_pitch, raw_yaw = output.cpu().numpy()[0]
        
        # Debug: Print raw model outputs (uncomment to debug)
        # print(f"Raw model output - Pitch: {raw_pitch:.6f} rad ({np.rad2deg(raw_pitch):.2f}°), Yaw: {raw_yaw:.6f} rad ({np.rad2deg(raw_yaw):.2f}°)")
        
        if apply_calibration:
            # Apply linear calibration (computed from gaze_calibration.json)
            # This maps MpiiFaceGaze outputs to DashGaze coordinate system
            raw_pitch_deg = np.rad2deg(raw_pitch)
            raw_yaw_deg = np.rad2deg(raw_yaw)
            
            calibrated_pitch_deg = 0.2550 * raw_pitch_deg + 8.4552
            calibrated_yaw_deg = 0.2396 * raw_yaw_deg + 1.6353
            
            # Convert back to radians
            pitch = np.deg2rad(calibrated_pitch_deg)
            yaw = np.deg2rad(calibrated_yaw_deg)
        else:
            # For Brain4Cars: The model was trained on MPIIGaze which has different camera setup
            # We need to use raw outputs but may need to scale/invert based on actual behavior
            # For now, use raw values directly - if they're clustered, it's likely a model limitation
            # or the model needs retraining/fine-tuning on Brain4Cars data
            
            # Use raw model outputs directly
            pitch = raw_pitch
            yaw = raw_yaw
            
            # If values are consistently clustered, it suggests the model outputs
            # don't vary much, which could be a model limitation for this dataset
        
        return pitch, yaw
