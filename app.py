import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

# Set Streamlit page configuration
st.set_page_config(page_title="Pomegranate Disease Detection", layout="centered")

# Load custom CSS (optional)
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css not found. Using default styling.")

# Load trained model
model = load_model('pomegranate_model.h5')

# Define class labels (must match model output)
class_names = ['Anthracnose', 'Bacterial_Spot', 'Normal', 'Rot', 'Scab']

# App title
st.markdown("<h1 style='text-align: center;'>🍎 Pomegranate Disease Detector</h1>", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload a pomegranate image...", type=["jpg", "jpeg", "png"])

# If file is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image_data.resize((128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict
    prediction = model.predict(img)

    if prediction.shape[1] != len(class_names):
        st.error(f"❌ Mismatch: Model predicted {prediction.shape[1]} classes but class_names has {len(class_names)}.")
    else:
        # Get prediction and confidence
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display prediction
        st.success(f"🩺 Prediction: **{predicted_class}** with **{confidence:.2f}%** confidence")

        # Show prediction probabilities
        probs = prediction[0]
        df = pd.DataFrame({
            'Class': class_names,
            'Confidence (%)': probs * 100
        })

        st.markdown("### 📊 Prediction Probabilities")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Class', sort='-y'),
            y='Confidence (%)',
            color='Class'
        )
        st.altair_chart(chart, use_container_width=True)
