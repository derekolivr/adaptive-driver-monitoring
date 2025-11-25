# ---------------------------------------------
# Gaze Estimation Training Pipeline (Cleaned)
# ---------------------------------------------
# This script loads the MPIIGaze dataset, trains a CNN-based model
# for 3D gaze estimation, evaluates its accuracy, and saves results.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

# Reproducibility
torch.manual_seed(42)
np.random_seed(42)


# Dataset

class MPIIGazeDataset(Dataset):
    """Loads normalized eye images, head poses, and gaze vectors."""

    def __init__(self, data_root, participants=None, days=None, transform=None):
        self.data_root = data_root
        self.transform = transform

        # Select participants
        if participants is None:
            participants = [f for f in os.listdir(data_root)
                            if os.path.isdir(os.path.join(data_root, f))]
        self.participants = participants

        # Storage
        self.images_left = []
        self.images_right = []
        self.gazes_left = []
        self.gazes_right = []
        self.head_poses_left = []
        self.head_poses_right = []

        # Load data from each participant
        for participant in self.participants:
            participant_path = os.path.join(data_root, participant)
            day_files = [f for f in os.listdir(participant_path) if f.endswith('.mat')]

            if days is not None:
                day_files = [f for f in day_files if f.split('.')[0] in days]

            for day_file in day_files:
                path = os.path.join(participant_path, day_file)
                try:
                    data = sio.loadmat(path)["data"]

                    left = data[0, 0]['left'][0, 0]
                    right = data[0, 0]['right'][0, 0]

                    # Append eye images and labels
                    self.images_left.append(np.expand_dims(left['image'], 1))
                    self.images_right.append(np.expand_dims(right['image'], 1))
                    self.gazes_left.append(left['gaze'])
                    self.gazes_right.append(right['gaze'])
                    self.head_poses_left.append(left['pose'])
                    self.head_poses_right.append(right['pose'])

                except Exception as e:
                    print(f"Error loading {path}: {e}")

        # Final concatenation
        self.images_left = np.vstack(self.images_left)
        self.images_right = np.vstack(self.images_right)
        self.gazes_left = np.vstack(self.gazes_left)
        self.gazes_right = np.vstack(self.gazes_right)
        self.head_poses_left = np.vstack(self.head_poses_left)
        self.head_poses_right = np.vstack(self.head_poses_right)

        print(f"Loaded {len(self.images_left)} samples from dataset.")

    def __len__(self):
        return len(self.images_left)

    def __getitem__(self, idx):
        left_img = self.images_left[idx].astype(np.float32) / 255.0
        right_img = self.images_right[idx].astype(np.float32) / 255.0

        left_img = torch.from_numpy(left_img)
        right_img = torch.from_numpy(right_img)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        eye_input = torch.cat([left_img, right_img], dim=0)

        gaze_target = torch.from_numpy(
            ((self.gazes_left[idx] + self.gazes_right[idx]) / 2).astype(np.float32)
        )
        head_pose = torch.from_numpy(
            ((self.head_poses_left[idx] + self.head_poses_right[idx]) / 2).astype(np.float32)
        )

        return {
            'eye_input': eye_input,
            'head_pose': head_pose,
            'gaze_target': gaze_target
        }



# Model

class GazeNet(nn.Module):
    """CNN for binocular eye-image based gaze estimation."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        flat_size = 128 * 4 * 7
        self.fc_eye = nn.Linear(flat_size, 256)
        self.fc_combined = nn.Linear(256 + 3, 128)
        self.fc_out = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, eye_input, head_pose):
        x = self.pool(self.relu(self.conv1(eye_input)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc_eye(x)))

        x = torch.cat([x, head_pose], dim=1)
        x = self.dropout(self.relu(self.fc_combined(x)))

        gaze = self.fc_out(x)
        gaze = gaze / torch.norm(gaze, dim=1, keepdim=True)

        return gaze



# Metrics


def angular_error(y_true, y_pred):
    y_true = y_true / torch.norm(y_true, dim=1, keepdim=True)
    y_pred = y_pred / torch.norm(y_pred, dim=1, keepdim=True)

    cos_sim = torch.clamp(torch.sum(y_true * y_pred, dim=1), -1.0, 1.0)
    angles = torch.acos(cos_sim) * 180.0 / np.pi
    return angles


# Training


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    history = {"train_loss": [], "val_loss": [], "train_angle": [], "val_angle": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, total_angle = 0, 0
        n_samples = 0

        for batch in train_loader:
            eye, pose, target = batch['eye_input'].to(device), batch['head_pose'].to(device), batch['gaze_target'].to(device)

            optimizer.zero_grad()
            pred = model(eye, pose)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            batch_size = eye.size(0)
            total_loss += loss.item() * batch_size
            total_angle += angular_error(target, pred).mean().item() * batch_size
            n_samples += batch_size

        train_loss = total_loss / n_samples
        train_angle = total_angle / n_samples

        # Validation
        model.eval()
        val_loss, val_angle, n_val = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                eye, pose, target = batch['eye_input'].to(device), batch['head_pose'].to(device), batch['gaze_target'].to(device)
                pred = model(eye, pose)
                loss = criterion(pred, target)

                batch_size = eye.size(0)
                val_loss += loss.item() * batch_size
                val_angle += angular_error(target, pred).mean().item() * batch_size
                n_val += batch_size

        val_loss /= n_val
        val_angle /= n_val

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_angle'].append(train_angle)
        history['val_angle'].append(val_angle)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Angle: {val_angle:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gaze_model.pth")

    model.load_state_dict(torch.load("best_gaze_model.pth"))
    return model, history



# Plots

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss']); plt.plot(history['val_loss'])
    plt.title('Loss'); plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_angle']); plt.plot(history['val_angle'])
    plt.title('Angular Error (deg)'); plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('training_history.png')



# Main

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_root = '/kaggle/input/mpiigaze/MPIIGaze/Data/Normalized'

    train_p = ['p07', 'p09', 'p06', 'p10']
    test_p = ['p12']

    train_dataset = MPIIGazeDataset(data_root, participants=train_p)
    test_dataset = MPIIGazeDataset(data_root, participants=test_p)

    train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)

    train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=128, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(train_dataset, val_idx), batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = GazeNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20)
    plot_training_history(history)

    model.eval()
    errors = []

    with torch.no_grad():
        for batch in test_loader:
            eye, pose, target = batch['eye_input'].to(device), batch['head_pose'].to(device), batch['gaze_target'].to(device)
            pred = model(eye, pose)
            errors.extend(angular_error(target, pred).cpu().numpy())

    print(f"Test Error: {np.mean(errors):.2f}° ± {np.std(errors):.2f}°")

    torch.save({
        'model_state_dict': model.state_dict(),
        'test_mean_angle': float(np.mean(errors)),
        'test_std_angle': float(np.std(errors))
    }, 'gaze_model_final.pth')


if __name__ == "__main__":
    main()
