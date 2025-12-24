import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quick Waste Classifier", layout="centered")
st.title("♻️ Waste Classifier ")
st.write("Upload an image to check if it's **Recyclable**, **Organic**, or **Non-Recyclable**.")


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
    image_data = image_data.convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = prediction[0][0]  # This is a number between 0 and 1

    return score


# --- HELPER FUNCTION TO SHOW RESULTS ---
def display_result(score):
    # LOGIC:
    # 0.00 - 0.40  --> Organic (Confident)
    # 0.40 - 0.60  --> Uncertain / Non-Recyclable (Confused)
    # 0.60 - 1.00  --> Recyclable (Confident)

    if score > 0.60:
        st.success(f"♻️ **RECYCLABLE** (Confidence: {score:.2f})")
        st.info("Place in the Green/Blue bin.")
        st.balloons()

    elif score < 0.40:
        st.success(f"🍎 **ORGANIC** (Confidence: {1 - score:.2f})")
        st.info("Place in the Compost/Organic bin.")

    else:
        # The "Gray Area" (Powerbanks, mixed trash, etc.)
        st.warning(f"⚠️ **NON-RECYCLABLE / UNCERTAIN**")
        st.write("The AI is not sure. This might be electronic waste or mixed trash.")
        st.caption(f"Confidence score: {score:.2f} (Too close to 50/50)")


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
        display_result(score)

# TAB 2: Live Camera
with tab2:
    st.header("Live Camera")
    cam_file = st.camera_input("Take a photo")

    if cam_file and model:
        image = Image.open(cam_file)
        st.image(image, caption="Captured Photo", use_column_width=True)
        st.write("Processing...")

        score = predict_waste(image)
        display_result(score)