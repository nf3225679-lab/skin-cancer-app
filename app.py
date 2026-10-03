import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()
height, width = model.input_shape[1:3]

st.title("🩺 Skin Cancer Detection")
st.write("Upload a skin lesion image and the model will predict its class.")
st.warning("Educational demo only. This is NOT a medical diagnosis - consult a dermatologist.")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = np.asarray(image.resize((width, height)), dtype="float32")[None, ...]   # raw 0-255 pixels
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))

    st.subheader(f"Prediction: {class_names[pred]}")
    st.metric("Confidence", f"{probs[pred] * 100:.2f}%")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))
