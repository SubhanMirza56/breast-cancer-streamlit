pip install tensorflow
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Load the trained model
model = load_model("breast_cancer_model.h5")

# Streamlit App UI
st.title("🧠 Breast Cancer Predictor")

# Create 30 input fields for the features
features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(30)]

# Predict button
if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0][0]
    result = "Benign (Non-Cancerous)" if prediction > 0.5 else "Malignant (Cancerous)"
    st.success(f"Prediction: **{result}**")
