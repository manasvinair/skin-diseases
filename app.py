import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/skin_disease_model.h5")
    return model

# Predict disease
def predict_disease(image, model, class_names):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

# Main app
def main():
    st.set_page_config(page_title="Skin Disease Classifier", layout="centered")


    st.title("Skin Disease Detection")
    st.write("Upload a skin image to identify the disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()

        class_names = [
            'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis',
            'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus', 'Moles',
            'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage',
            'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 'Vasculitis', 'Vitiligo', 'Warts'
        ]

        predicted_class, confidence = predict_disease(image, model, class_names)

        st.success(f"**Predicted Disease:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2%}")

if __name__ == "__main__":
    main()
