import streamlit as st
import tensorflow as tf
import numpy as np
import keras
# import cv2
from keras.models import load_model
from PIL import Image

MODEL_PATH = "model/k_fold_model.h5"
model = load_model(MODEL_PATH, custom_objects={})
classes = ["Normal","Cataract","Diabetes","Glaucoma","Hypertension","Myopia","Age_Issues","Other"]

def preprocess_image(image):
    # Preprocess the image as needed (resize, normalization, etc.)
    # Example:
    resized_image = image.resize((224, 224))
    array_image = np.array(resized_image)
    normalized_image = array_image / 255.0  # Normalize pixel values
    return normalized_image

im = Image.open('elements/logo.jpeg')
st.set_page_config(
    page_title="CyberRetina",
    page_icon=im,
)
st.image('elements/logo.jpeg', use_column_width=True)
st.header("Our mission: ")
st.markdown("##### Enhancing Ophthalmology Patient Management through Intelligent Recognition of Ocular Diseases Using Deep Learning and Fundus Image Analysis")
st.write("")
st.subheader("Output is  among the following: ")
st.markdown(" ##### [Normal, Cataract, Diabetes, Glaucoma, Hypertension, Myopia, Age Issues, Others]")
st.write("")
st.write("")
image_file = st.file_uploader("Upload the image of your Retina here: ", type=[".jpg", ".png", ".jpeg", ".pdf"])
st.write("")
if st.button("Predict"):
    if image_file:
        image = Image.open(image_file)
        st.write('')
        preprocessed_image = preprocess_image(image)

        image_expand = np.expand_dims(preprocessed_image, axis=0)

        predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_class = np.argmax(predictions)
        st.success(f'Prediction: {classes[predicted_class]}')
        st.image(image, caption='Uploaded Image', use_column_width=True)

    else:
        st.error("No file selected")