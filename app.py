"""Streamlit inference demo for face-mask classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

CLASS_NAMES = ["incorrect_mask", "with_mask", "without_mask"]
IMAGE_SIZE = (128, 128)

st.set_page_config(page_title="Face Mask Transfer Learning", layout="wide")
st.title("Face Mask Detection with Transfer Learning")
st.caption("Three-class image classifier demo: with mask, without mask, and mask worn incorrectly.")

@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv("data/processed/model_comparison.csv")

@st.cache_data
def load_classification_report() -> pd.DataFrame:
    return pd.read_csv("data/processed/vgg16_classification_report.csv")

comparison = load_model_comparison()
report = load_classification_report()

best = comparison.sort_values("test_accuracy", ascending=False).iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Best Model", best["model"])
c2.metric("Test Accuracy", f"{best['test_accuracy']:.2%}")
c3.metric("Classes", len(CLASS_NAMES))

st.subheader("Model Comparison")
st.dataframe(comparison, use_container_width=True)

st.subheader("Per-Class Classification Report")
st.dataframe(report, use_container_width=True)

st.subheader("Inference Demo")
st.write(
    "Upload a locally trained `.keras` or `.h5` model and a face image to run inference. "
    "The repository does not include trained model weights or the full image dataset, so the demo is designed for local reproduction."
)

uploaded_model = st.file_uploader("Optional: upload trained Keras model", type=["keras", "h5"])
uploaded_image = st.file_uploader("Upload image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", width=320)

    image_array = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if uploaded_model is None:
        st.info("Upload trained model weights to enable live prediction. The image preprocessing preview is ready.")
    else:
        try:
            import tensorflow as tf

            model_path = Path("/tmp/uploaded_face_mask_model.keras")
            model_path.write_bytes(uploaded_model.read())
            model = tf.keras.models.load_model(model_path)
            probabilities = model.predict(image_array)[0]
            prediction_index = int(np.argmax(probabilities))
            prediction = CLASS_NAMES[prediction_index]

            st.success(f"Predicted class: {prediction}")
            st.bar_chart(pd.DataFrame({"probability": probabilities}, index=CLASS_NAMES))
        except Exception as exc:  # pragma: no cover - Streamlit runtime guard
            st.error(f"Unable to load model or run prediction: {exc}")

st.subheader("Responsible Use")
st.write(
    "This demo is for portfolio and educational use. Any real-world deployment should be validated on representative images, "
    "monitored for drift, and reviewed for fairness and privacy risks."
)
