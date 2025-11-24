import cv2
import pandas as pd
import os
from tqdm import tqdm
import json
import numpy as np # Import numpy directly

def process_dashgaze_data(video_path, csv_path, output_dir):
    """
    Processes the DashGaze video and CSV to extract synchronized, cropped image pairs
    and their corresponding ground truth data.
    """
    # 1. Load the CSV data, handling potential malformed lines
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"Error reading CSV: {e}")
        print("Attempting to read with a different parser...")
        df = pd.read_csv(csv_path, sep=',', engine='python', on_bad_lines='skip')

    # Remove rows with non-numeric frame numbers if any
    df = df[pd.to_numeric(df['dash_frames'], errors='coerce').notna()]
    df['dash_frames'] = df['dash_frames'].astype(int)

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded successfully. Total frames: {total_frames_in_video}")

    # --- MODIFICATION: Sample 10 frames spread evenly across the dataset ---
    num_samples = 10
    indices = np.linspace(0, len(df) - 1, num_samples, dtype=int)
    df_sampled = df.iloc[indices]
    print(f"Sampling {num_samples} frames evenly spaced throughout the video.")

    # 3. Define crop regions using the exact coordinates you provided
    # --- FIX: Using pixel-perfect coordinates from the find_dimensions.py tool ---
    driver_crop = (6, 7, 950, 1068)      # x, y, width, height
    road_crop = (962, 6, 955, 1071)      # x, y, width, height

    # 4. Loop through the CSV and extract frames
    print(f"Processing {len(df_sampled)} frames...")
    for index, row in tqdm(df_sampled.iterrows(), total=df_sampled.shape[0], desc="Extracting Frames"):
        frame_number = row['dash_frames']
        
        if frame_number >= total_frames_in_video:
            print(f"Warning: Frame number {frame_number} in CSV is out of video bounds. Skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            # Crop the driver and road views
            driver_view = frame[driver_crop[1]:driver_crop[1]+driver_crop[3], driver_crop[0]:driver_crop[0]+driver_crop[2]]
            road_view = frame[road_crop[1]:road_crop[1]+road_crop[3], road_crop[0]:road_crop[0]+road_crop[2]]

            # Save the cropped images
            base_filename = f"frame_{frame_number:05d}"
            cv2.imwrite(os.path.join(output_dir, f"{base_filename}_driver.jpg"), driver_view)
            cv2.imwrite(os.path.join(output_dir, f"{base_filename}_road.jpg"), road_view)
            
            # --- NEW: Save ground truth data as a JSON file ---
            ground_truth = {
                'azimuth_deg': row.get('azimuth [deg]'),
                'elevation_deg': row.get('elevation [deg]'),
                # Add any other relevant ground truth columns here
            }
            with open(os.path.join(output_dir, f"{base_filename}_gt.json"), 'w') as f:
                json.dump(ground_truth, f, indent=4)

    # 5. Release the video capture object
    cap.release()
    print("Processing complete.")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Before running, update this path to point to your DashGaze video file.
    video_file = os.path.join("test", "DashGaze_test.mp4") 
    
    csv_file = os.path.join("test", "DashGaze_sensors_selected.csv")
    output_directory = os.path.join("test_data", "dashgaze_processed")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
        
    if not os.path.exists(video_file):
        print("="*50)
        print("ERROR: Video file not found!")
        print(f"Please update the 'video_file' variable in '{__file__}'")
        print("to the correct path for your DashGaze video file.")
        print("="*50)
    else:
        process_dashgaze_data(video_file, csv_file, output_directory)
