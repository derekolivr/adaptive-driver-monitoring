"""
Fine-tune the MpiiFaceGaze model on DashGaze dataset for better coordinate alignment.

This script:
1. Loads the pre-trained gaze_tracker_endterm.pth model
2. Freezes early ResNet18 layers (feature extraction)
3. Retrains final layers on DashGaze data
4. Saves the fine-tuned model as gaze_tracker_dashgaze.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import glob
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gaze_model import get_gaze_model


class DashGazeDataset(Dataset):
    """Dataset class for DashGaze preprocessed images with ground truth."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Find all ground truth files
        gt_files = sorted(glob.glob(os.path.join(data_dir, "*_gt.json")))
        
        print(f"Loading DashGaze dataset from {data_dir}...")
        for gt_path in gt_files:
            base_name = os.path.basename(gt_path).replace("_gt.json", "")
            driver_path = os.path.join(data_dir, f"{base_name}_driver.jpg")
            
            if os.path.exists(driver_path):
                # Load ground truth
                with open(gt_path, 'r') as f:
                    gt = json.load(f)
                
                # Convert to radians (dataset is in degrees)
                azimuth_rad = np.deg2rad(gt['azimuth_deg'])
                elevation_rad = np.deg2rad(gt['elevation_deg'])
                
                # Store as (image_path, [pitch, yaw]) - matching MpiiFaceGaze format
                # We map DashGaze elevation -> pitch, azimuth -> yaw
                gaze_angles = np.array([elevation_rad, azimuth_rad], dtype=np.float32)
                self.samples.append((driver_path, gaze_angles))
        
        print(f"Loaded {len(self.samples)} samples from DashGaze dataset.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, gaze_angles = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(gaze_angles, dtype=torch.float32)


def finetune_model(pretrained_model_path, data_dir, output_path, epochs=10, freeze_layers=True):
    """
    Fine-tune a pre-trained gaze model on DashGaze data.
    
    Args:
        pretrained_model_path: Path to the pre-trained .pth file
        data_dir: Directory containing preprocessed DashGaze images
        output_path: Where to save the fine-tuned model
        epochs: Number of training epochs
        freeze_layers: If True, freeze ResNet18 backbone (only train final layers)
    """
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS for GPU acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    
    # 1. Load pre-trained model
    print(f"\nLoading pre-trained model from {pretrained_model_path}...")
    model = get_gaze_model()
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.to(device)
    
    # 2. Optionally freeze early layers
    if freeze_layers:
        print("Freezing ResNet18 backbone layers (keeping feature extraction)...")
        # Freeze all layers except the final fc layers
        for name, param in model.named_parameters():
            if 'fc' not in name:  # fc is the final classification layer we defined
                param.requires_grad = False
        print("Only the final gaze prediction layers will be trained.")
    else:
        print("Training all layers (full fine-tuning).")
    
    # 3. Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    full_dataset = DashGazeDataset(data_dir=data_dir, transform=transform)
    
    if len(full_dataset) == 0:
        print("ERROR: No samples found in the dataset. Please run preprocess_dashgaze.py first.")
        return
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\nDataset split: {train_size} training, {val_size} validation samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 4. Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    best_val_loss = float('inf')
    best_model_path = output_path.replace('.pth', '_best.pth')
    
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print("=" * 70)
    
    # 5. Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, gaze_angles in train_loader:
            images = images.to(device)
            gaze_angles = gaze_angles.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gaze_angles)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, gaze_angles in val_loader:
                images = images.to(device)
                gaze_angles = gaze_angles.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, gaze_angles)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.6f})")
    
    print("=" * 70)
    print(f"\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    
    # Save final model as well
    torch.save(model.state_dict(), output_path)
    print(f"Final model saved to: {output_path}")


if __name__ == '__main__':
    # Configuration
    PRETRAINED_MODEL = "gaze_tracker_endterm.pth"
    DATA_DIR = os.path.join("test_data", "dashgaze_processed")
    OUTPUT_MODEL = "gaze_tracker_dashgaze_finetuned.pth"
    EPOCHS = 15
    FREEZE_BACKBONE = True  # Set to False for full fine-tuning
    
    print("=" * 70)
    print("DashGaze Fine-Tuning Script")
    print("=" * 70)
    print(f"Pre-trained model: {PRETRAINED_MODEL}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output model: {OUTPUT_MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print("=" * 70)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Data directory '{DATA_DIR}' not found!")
        print("Please run 'python scripts/preprocess_dashgaze.py' first to generate training data.")
        sys.exit(1)
    
    # Run fine-tuning
    finetune_model(
        pretrained_model_path=PRETRAINED_MODEL,
        data_dir=DATA_DIR,
        output_path=OUTPUT_MODEL,
        epochs=EPOCHS,
        freeze_layers=FREEZE_BACKBONE
    )

