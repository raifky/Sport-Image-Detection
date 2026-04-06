import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

def run():

    st.title("🔮 Model Inference")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Cek 2 kemungkinan lokasi model
    model_path = os.path.join(BASE_DIR, "sports_classification_model.keras")

    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "..", "sports_classification_model.keras")

    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(model_path)

    model = load_model()

    class_names = ['basketball', 'boxing', 'baseball', 'tennis', 'judo']

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        img = image.resize((224,224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        threshold = 0.6

        st.subheader("Prediction Result")

        if confidence < threshold:
            st.warning("⚠ Model tidak yakin dengan prediksi.")
        else:
            st.success(f"Predicted Class: {predicted_class}")

        st.write("Confidence:", round(confidence*100, 2), "%")

        st.subheader("Probability Distribution")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {round(float(prob)*100,2)}%")
