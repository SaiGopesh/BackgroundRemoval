import streamlit as st
from roboflow import Roboflow
from PIL import Image
import requests
import io

# Set page config
st.set_page_config(layout="wide", page_title="Animal Classification")

st.write("## Classify your animal image")
st.write(
    "Upload an image to classify the animal in it using a pre-trained model from Roboflow. The model is trained on a variety of animals and will predict the class of the animal. :elephant: :lion: :dog: :cat:"
)
st.sidebar.write("## Upload your image :gear:")

# Maximum file size for upload
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Roboflow API configuration (replace with your Roboflow API key and project details)
rf = Roboflow(api_key="YjMxjvei1qSX2MwmTqkv")  # Replace with your API key
project = rf.workspace("ttu-py3sj").project("animal-classification-rextd")
model = project.version(1).model  # Load the model

# Function to predict the animal using Roboflow API
def classify_animal(upload):
    image = Image.open(upload)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to byte array for Roboflow API
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    byte_arr = byte_arr.getvalue()

    # Send the image to the Roboflow API for prediction
    response = model.predict(byte_arr, confidence=40, overlap=30).json()

    # Display the prediction results
    if response['predictions']:
        prediction = response['predictions'][0]  # Get the top prediction
        predicted_class = prediction['class']
        confidence = prediction['confidence']
        
        st.write(f"Prediction: {predicted_class} with confidence: {confidence*100:.2f}%")
    else:
        st.write("No prediction could be made. Please try again.")

    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download the image", upload, "image.png", "image/png")

# Upload the image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        classify_animal(upload=uploaded_file)
else:
    st.write("Please upload an image to classify.")
