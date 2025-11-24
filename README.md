# Adaptive Driver Monitoring System

This project is an advanced driver assistance system that integrates multiple AI models to provide a comprehensive analysis of a driver's attention and the surrounding road conditions. It uses a combination of gaze tracking, driver distraction classification, and road object detection to assess the overall driving situation in real-time.

## Key Features

- **Multi-Model Integration**: Combines three distinct neural networks for a holistic understanding of driver behavior and context.
- **Gaze Tracking**: Utilizes a fine-tuned ResNet18 model to predict the driver's gaze direction (pitch and yaw), determining if they are looking at the road, mirrors, or center console.
- **Distraction Classification**: Employs a VGG-based model to classify the driver's actions, identifying behaviors such as texting, talking on the phone, or operating the radio.
- **Road Context Analysis**: Uses a pre-trained YOLOv8 model to detect critical objects in the road ahead, including other vehicles, pedestrians, and traffic signals.
- **Fusion Engine**: A rule-based engine that intelligently combines the outputs from all models to provide a final, high-level assessment of the driver's state (e.g., "Focused," "Distracted," "Warning").
- **Interactive Dashboard**: A user-friendly web interface built with Streamlit that visualizes the outputs from all models.

## Models Used

- **Gaze Tracking**: A **ResNet18** model, pre-trained on ImageNet and fine-tuned on the MPIIGaze dataset to predict gaze direction (pitch and yaw).
- **Distraction Classification**: A **VGG-based Convolutional Neural Network** (`driver_distraction_model_vgg.h5`) trained to classify various driver behaviors and states of distraction.
- **Road Context Analysis**: A pre-trained **YOLOv8n** (nano) model from Ultralytics, used for real-time object detection of relevant entities such as cars, pedestrians, and traffic signals.

## Project Structure

```
adaptive-driver-monitoring/
│
├── Driver_Gaze_Map/      # Original scripts and notebooks for training the gaze model
│   ├── dataset.py
│   ├── train.py
│   └── ...
│
├── Midterm/              # Contains the pre-trained driver distraction model
│   └── driver_distraction_model_vgg.h5
│
├── src/                  # Source code for the main application
│   ├── app.py            # The main Streamlit dashboard application
│   ├── gaze_model.py     # Gaze model architecture (ResNet18)
│   ├── gaze_tracker.py   # Module to load and run the gaze model
│   ├── distraction_classifier.py # Module for the VGG distraction model
│   ├── road_context.py   # Module for YOLOv8 road object detection
│   └── fusion_engine.py  # Core logic for combining model outputs
│
├── data/                 # (If present) Holds the training data, like MPIIGaze
├── gaze_tracker_endterm.pth # The final trained gaze tracker model weights
└── requirements.txt      # Python dependencies for the project
```

## Setup and Installation

1.  **Clone the repository**:

    ```bash
    git clone <your-repo-url>
    cd adaptive-driver-monitoring
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Once the setup is complete, you can launch the Streamlit dashboard by running the following command from the project's root directory:

```bash
streamlit run src/app.py
```

This will open the application in your web browser. You can then upload an image of a driver and an image of the road to see the full analysis.

## Future Plans

The next major step for this project is to transition from a file-based proof-of-concept to a real-time, camera-based system. The plan includes:

- **Real-Time Video Processing**: Modify the Streamlit app to accept live video feeds from two webcams (one facing the driver, one facing the road).
- **Continuous Analysis**: The system will run the models on the video streams continuously, providing a live assessment of the driver's state.
- **Alert System**: Implement a more dynamic alert system that tracks the _duration_ of a distraction. For example, glancing at the radio for 1 second is acceptable, but looking for 5 seconds will trigger a high-priority alert.
- **Performance Optimization**: Investigate model optimization techniques (like quantization or pruning) and efficient video processing pipelines to ensure the system can run smoothly in real-time on consumer-grade hardware.
