import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quick Waste Classifier", layout="centered")
st.title("♻️ Waste Classifier ")
st.write("Upload an image to check if it's **Recyclable** or **Organic**.")


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'waste_model_small.keras'

    if not os.path.exists(model_path):
        st.error(f"❌ Error: File '{model_path}' not found. Please download it from Colab and put it in this folder.")
        return None
    return tf.keras.models.load_model(model_path)


model = load_model()


# --- PREDICTION FUNCTION ---
def predict_waste(image_data):
    # 1. Convert to RGB (Fixes the transparency error!)
    image_data = image_data.convert('RGB')

    # 2. Resize to 224x224
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)

    # 3. Convert to array
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predict
    prediction = model.predict(img_array)
    return prediction[0][0]


# --- INTERFACE ---
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])

# TAB 1: Upload
with tab1:
    st.header("Upload a Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        score = predict_waste(image)

        if score > 0.5:
            st.success(f"♻️ **RECYCLABLE** (Confidence: {score:.2f})")
        else:
            st.success(f"🍎 **ORGANIC** (Confidence: {1 - score:.2f})")

# TAB 2: Live Camera (NOW ENABLED!)
with tab2:
    st.header("Live Camera")

    # I have fixed the indentation here for you:
    cam_file = st.camera_input("Take a photo")

    if cam_file and model:
        image = Image.open(cam_file)
        st.image(image, caption="Captured Photo", use_column_width=True)
        st.write("Processing...")

        score = predict_waste(image)

        if score > 0.5:
            st.success(f"♻️ **RECYCLABLE** (Confidence: {score:.2f})")
        else:
            st.success(f"🍎 **ORGANIC** (Confidence: {1 - score:.2f})")