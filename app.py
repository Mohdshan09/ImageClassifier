import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained VGG16 model + higher level layers
model = VGG16(weights='imagenet')

# Streamlit UI
st.title("Object Identification using VGG16")
st.write("Upload an image and the model will identify objects in it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Pre-process the image to fit the VGG16 model input size
    image = image.resize((224, 224))  # VGG16 requires 224x224 input
    image_array = img_to_array(image)  # Convert image to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess the image for VGG16

    # Predict using the VGG16 model
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode the top 3 predictions

    # Show predictions
    st.write("Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. {label}: {score:.4f}")
