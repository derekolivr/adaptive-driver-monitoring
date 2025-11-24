import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import get_gaze_model
from dataset import GazeDataset
from torchvision import transforms
import os

# Example of how you would use GazeDataset:
# When the real data is ready, change use_dummy_data to False
# and ensure the data_dir points to the extracted MPIIGaze folder.
DATA_DIR = os.path.join('data', 'MPIIGaze')
USE_DUMMY_DATA = False

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Initialize the full dataset
    print("Loading dataset...")
    full_dataset = GazeDataset(data_dir=DATA_DIR, transform=transform, use_dummy_data=USE_DUMMY_DATA)
    
    # Split dataset into training and validation
    val_split = 0.1
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Found {dataset_size} samples. Splitting into {train_size} training and {val_size} validation samples.")

    # Dynamically set num_workers for optimal performance
    num_cpu_cores = os.cpu_count() or 2 # Default to 2 if os.cpu_count() is None
    print(f"Using {num_cpu_cores} CPU cores for data loading.")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_cpu_cores)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_cpu_cores)

    # Start training with early stopping
    train_model(train_loader, val_loader, epochs=10, patience=2)


def train_model(train_loader, val_loader, epochs=10, patience=2):
    # Set device to Apple's MPS for GPU acceleration on Mac, otherwise fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")
    
    # Initialize Model
    model = get_gaze_model().to(device)
    
    # Loss: MSE because we are predicting continuous coordinates (angles)
    criterion = nn.MSELoss() 
    
    # Optimizer: Adam, consistent with your midterm methodology 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Starting Training...")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "gaze_tracker_best.pth"

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)

            # ===== Sanity Check for the first batch =====
            if epoch == 0 and i == 0:
                print("\n--- Sanity Check ---")
                print(f"Image batch shape:  {images.shape}")
                print(f"Label batch shape:  {labels.shape}")
                print(f"Model output shape: {outputs.shape}")
                print("--------------------\n")
            # ============================================
            
            loss = criterion(outputs, labels.float())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Training Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saving best model to {best_model_path}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Save the best model with the final name for your demo
    final_model_path = "gaze_tracker_endterm.pth"
    # Ensure the best model was saved before trying to rename
    if os.path.exists(best_model_path):
        os.rename(best_model_path, final_model_path)
        print(f"Training finished. Best model saved to {final_model_path}")
    else:
        print("Training finished, but no improvement was seen over the initial state. No model saved.")

if __name__ == '__main__':
    main()

# Note: You will need to write the 'GazeDataset' class to load your specific 
# downloaded images, similar to how you loaded the State Farm data[cite: 36].