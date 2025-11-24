import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import scipy.io as sio
import glob
import numpy as np

class GazeDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_dummy_data=True):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        if use_dummy_data:
            print("--- Using Dummy Data ---")
            # Create a few dummy samples for testing purposes
            self.samples = [
                ('dummy_path_1.jpg', np.array([0.1, 0.1])),
                ('dummy_path_2.jpg', np.array([-0.2, 0.3])),
                ('dummy_path_3.jpg', np.array([0.0, -0.15]))
            ]
            self.use_dummy_data = True
            return
        
        self.use_dummy_data = False
        
        # The real implementation to parse the dataset
        person_folders = glob.glob(os.path.join(data_dir, 'Data', 'Normalized', 'p*'))
        
        print(f"Found {len(person_folders)} participants. Processing...")
        for person_folder in person_folders:
            participant_id = os.path.basename(person_folder)
            print(f"  - Processing participant: {participant_id}")
            annotation_files = glob.glob(os.path.join(person_folder, '*.mat'))
            
            for ann_file in annotation_files:
                day_id = os.path.splitext(os.path.basename(ann_file))[0]
                image_base_dir = os.path.join(self.data_dir, 'Data', 'Original', participant_id, day_id)
                
                mat_contents = sio.loadmat(ann_file)
                
                # Extract filenames and gaze data.
                # The gaze data is nested under 'data' -> 'right' -> 'gaze' or 'left' -> 'gaze'
                filenames = mat_contents['filenames']
                # We'll use the right eye's gaze data.
                gaze_data = mat_contents['data']['right'][0, 0]['gaze'][0, 0]
                
                for i, filename in enumerate(filenames):
                    # The filename is nested in the .mat file's cell array structure
                    img_filename = filename[0][0].strip()
                    img_path = os.path.join(image_base_dir, img_filename)
                    gaze_vector_3d = gaze_data[i] # This is a 3D vector [x, y, z]
                    
                    # Convert the 3D gaze vector to 2D pitch and yaw angles
                    x, y, z = gaze_vector_3d
                    pitch = np.arcsin(-y)
                    yaw = np.arctan2(-x, -z)
                    
                    # We only need the pitch and yaw for our model
                    gaze_vector_2d = np.array([pitch, yaw]).astype(np.float32)
                    self.samples.append((img_path, gaze_vector_2d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, gaze_label = self.samples[idx]
        
        if self.use_dummy_data:
            # For dummy data, create a random image tensor
            image = torch.randn(3, 224, 224)
        else:
            image = Image.open(img_path).convert('RGB')

        gaze_tensor = torch.tensor(gaze_label, dtype=torch.float32)

        if self.transform and not self.use_dummy_data:
            image = self.transform(image)
            
        return image, gaze_tensor
