import torch
import torch.nn as nn
import torchvision.models as models

def get_gaze_model():
    # 1. Load a pre-trained ResNet18 (Lightweight and fast)
    model = models.resnet18(pretrained=True)
    
    # 2. Modify the input layer
    # ResNet expects 3 channels (RGB)
    
    # 3. Modify the Output Layer (The "Head")
    # Original ResNet has 1000 classes. We need 2 outputs: [Pitch, Yaw]
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),  # Regularisation
        nn.Linear(256, 2) # Output: 2 Coordinates (x, y)
    )
    
    return model

# Check if it runs
if __name__ == "__main__":
    net = get_gaze_model()
    print("Gaze Model Architecture Ready.")
