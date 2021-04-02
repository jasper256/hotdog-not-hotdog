from keras import models
import numpy as np
import streamlit as st
from cv2 import resize
from PIL import Image


IMG_SIZE = 100 # Must match most recent train data
model = models.load_model("hotdog-not-hotdog-classifier")

def is_hotdog(file):
    # Clean input image, feed it through the model, and return prediction
    img = Image.open(file)
    img_array = np.array(img)
    formatted = resize(img_array, (IMG_SIZE, IMG_SIZE))
    formatted = formatted / 255
    prediction = model.predict(np.array([formatted,]))
    return bool(int(prediction[0] + .5)) # Prediction will be closer to 0 for "not hotdog" and closer to 1 for "hotdog"


st.title("Hotdog or Not Hotdog?")

file = st.file_uploader("Upload picture of hotdog or not hotdog.", type="jpg")
if file is not None:
    st.image(file)
    if is_hotdog(file):
        st.write("Processing Complete: Hotdog")
    else:
        st.write("Processing Complete: Not Hotdog")

st.write("View the code on GitHub: https://github.com/jasper256/hotdog-not-hotdog")