import tensorflow as tf
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
        self.model = tf.keras.models.load_model(model_path)
        print("Distraction Classifier model loaded.")
        
        # Standard class labels for driver distraction datasets
        self.class_labels = [
            'c0: safe driving',
            'c1: texting - right',
            'c2: talking on the phone - right',
            'c3: texting - left',
            'c4: talking on the phone - left',
            'c5: operating the radio',
            'c6: drinking',
            'c7: reaching behind',
            'c8: hair and makeup',
            'c9: talking to passenger'
        ]
        
        # Get the expected input shape from the model
        self.input_shape = self.model.input_shape[1:3] # (height, width)

    def classify_distraction(self, driver_image: Image.Image):
        """
        Classifies the driver's action from a PIL image.
        """
        driver_image = driver_image.convert("RGB")
        # Resize to the model's expected input size
        image_resized = driver_image.resize(self.input_shape)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_resized)
        image_array = image_array / 255.0
        
        # Add a batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Make a prediction
        predictions = self.model.predict(image_batch)
        
        # Get the top prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class_label = self.class_labels[predicted_class_index]
        
        return predicted_class_label, confidence
