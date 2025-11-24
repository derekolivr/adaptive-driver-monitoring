import tensorflow as tf
import keras # Import keras as a top-level package
from PIL import Image
import numpy as np
import os

class DistractionClassifier:
    def __init__(self, model_path=os.path.join("Midterm", "driver_distraction_model_vgg.h5")):
        """
        Initializes the DistractionClassifier model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Distraction model file not found at {model_path}")
        
        print("Loading Distraction Classifier model...")
        self.model = keras.models.load_model(model_path)
        print("Distraction Classifier model loaded.")
        
        # Standard class labels for driver distraction datasets
        self.class_labels = [
            "c0: safe driving",
            "c1: texting - right",
            "c2: talking on the phone - right",
            "c3: texting - left",
            "c4: talking on the phone - left",
            "c5: operating the radio",
            "c6: drinking",
            "c7: reaching behind",
            "c8: hair and makeup",
            "c9: talking to passenger"
        ]
        
        # Get input shape from the model
        model_input_shape = self.model.input_shape
        
        # Handle cases where the model has undefined input shape (None, None)
        if model_input_shape[1] is None or model_input_shape[2] is None:
            print("Warning: Model has undefined input shape. Falling back to (224, 224).")
            self.input_shape = (224, 224)
        else:
            self.input_shape = model_input_shape[1:3]
        
        print(f"DistractionClassifier initialized with input shape: {self.input_shape}")
    
    def classify_distraction(self, driver_image: Image.Image):
        """
        Classifies the driver's distraction state from a driver-facing image.
        
        Args:
            driver_image: PIL Image of the driver
            
        Returns:
            tuple: (predicted_class_label, confidence)
        """
        # Convert to RGB if needed
        driver_image = driver_image.convert("RGB")
        
        # Resize to model's expected input size
        image_resized = driver_image.resize(self.input_shape)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_resized) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Run prediction
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_class_label = self.class_labels[predicted_class_index]
        
        return predicted_class_label, confidence