import torch
from torchvision import transforms
from PIL import Image
from model import get_gaze_model
import os
from dataset import GazeDataset # Import the dataset class
import numpy as np # Import numpy for calculations

def predict_gaze(image_path, model_path="gaze_tracker_best.pth"):
    """
    Predicts the gaze coordinates from a single image.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained .pth model file.

    Returns:
        tuple: A tuple containing the predicted (pitch, yaw) angles.
    """
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")

    # 1. Load the model architecture and then the saved weights
    model = get_gaze_model()
    
    # Check if the model file exists before trying to load
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 2. Set the model to evaluation mode
    # This is important because it disables layers like Dropout
    model.eval()

    # 3. Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 4. Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
        
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    image_tensor = image_tensor.to(device)

    # 5. Make a prediction
    with torch.no_grad(): # No need to calculate gradients for inference
        output = model(image_tensor)
    
    # The output is a tensor, get the values
    pitch, yaw = output.cpu().numpy()[0]
    
    return pitch, yaw

if __name__ == '__main__':
    # --- How to test on an image from the dataset ---

    # 1. Load the dataset (make sure use_dummy_data is False to load real data)
    print("Loading the dataset to select a test image...")
    # This path is correct when you run the script from the root project folder
    DATA_DIR = os.path.join('data', 'MPIIGaze') 
    full_dataset = GazeDataset(data_dir=DATA_DIR, use_dummy_data=False)

    if len(full_dataset) > 0:
        # 2. Pick a sample from the dataset
        # You can change the index (e.g., 0, 10, 100) to test different images
        sample_index = 100
        image_path, true_gaze = full_dataset.samples[sample_index]

        print(f"\nTesting with sample #{sample_index} from the dataset.")
        print(f"Image path: {image_path}")
        
        # 3. Run the prediction
        # The model path is also relative to the root folder
        predicted_gaze = predict_gaze(image_path, model_path="gaze_tracker_best.pth")
        
        # 4. Compare the results
        if predicted_gaze:
            pred_pitch, pred_yaw = predicted_gaze
            true_pitch, true_yaw = true_gaze
            
            # --- Calculate and display the error ---
            pitch_error = abs(pred_pitch - true_pitch)
            yaw_error = abs(pred_yaw - true_yaw)
            
            # Calculate Euclidean distance as an overall error metric
            euclidean_distance = np.sqrt(pitch_error**2 + yaw_error**2)
            
            print(f"\nPrediction successful!")
            print(f"  - Predicted Pitch: {pred_pitch:.4f}  |  True Pitch: {true_pitch:.4f}  |  Error: {pitch_error:.4f}")
            print(f"  - Predicted Yaw:   {pred_yaw:.4f}  |  True Yaw:   {true_yaw:.4f}  |  Error: {yaw_error:.4f}")
            print(f"  - Overall Error (Euclidean Distance): {euclidean_distance:.4f}")

    else:
        print("Dataset is empty or could not be loaded. Please check the path and data.")