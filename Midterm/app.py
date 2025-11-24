import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Adaptive Driver Monitoring", page_icon="🚗")

st.title("Driver Distraction Detection")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose a model", ["Custom Model", "VGG16 Model"])
input_choice = st.sidebar.selectbox("Choose input type", ["Image", "Webcam"])

if model_choice == "Custom Model":
    model_path = "driver_distraction_model_custom.h5"
else:
    model_path = "driver_distraction_model_vgg.h5"

try:
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

class_names = [
    'safe driving', 
    'texting - right', 
    'talking on the phone - right', 
    'texting - left', 
    'talking on the phone - left', 
    'operating the radio', 
    'drinking', 
    'reaching behind', 
    'hair and makeup', 
    'talking to passenger'
]

def preprocess_image(image, model):
    input_shape = model.input_shape
    target_h, target_w, channels = input_shape[1], input_shape[2], input_shape[3]
    
    if channels == 1 and image.mode != "L":
        image = image.convert("L")
    elif channels == 3 and image.mode != "RGB":
        image = image.convert("RGB")
        
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h))

    image_array = np.array(image)
    if channels == 1:
        image_array = np.expand_dims(image_array, axis=-1)
        
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0


if input_choice == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        processed_image = preprocess_image(image, model)
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction)

        st.success(f"Prediction: **{predicted_class_name}** with {confidence:.2f} confidence.")

elif input_choice == "Webcam":
    st.write("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        st.stop()
    
    frame_placeholder = st.empty()

    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended.")
            break
        
        # Convert the frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Preprocess and predict
        processed_image = preprocess_image(image, model)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"{predicted_class_name} ({confidence:.2f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
