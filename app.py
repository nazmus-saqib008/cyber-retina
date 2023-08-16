import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

MODEL_PATH= "model/k_fold_model.h5"
model = load_model(MODEL_PATH)
classes = ["Normal","Cataract","Diabetes","Glaucoma","Hypertension","Myopia","Age_Issues","Other"]


def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])

    return classes[predicted_class]

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
        prediction = predict(image)
        st.success(f'Prediction: {prediction}')
        st.image(image, caption='Uploaded Image',use_column_width=True)

    else:
        st.error("No file selected")