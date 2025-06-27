import streamlit as st
import numpy as np
import joblib

# Load trained objects
model = joblib.load('star_model.pkl')
scaler = joblib.load('scaler.pkl')
le_color = joblib.load('le_color.pkl')
le_spectral = joblib.load('le_spectral.pkl')

# Title
st.title("Star Type Classifier")
st.write("Enter stellar properties to predict the type of star")

# Input fields
temperature = st.number_input("Temperature (K)", min_value=1000, max_value=50000, value=5778)
luminosity = st.number_input("Luminosity (L/Lo)", value=1.0)
radius = st.number_input("Radius (R/Ro)", value=1.0)
absolute_magnitude = st.number_input("Absolute Magnitude", value=5.0)

star_color = st.selectbox("Star Color", le_color.classes_)
spectral_class = st.selectbox("Spectral Class", le_spectral.classes_)

if st.button("Predict Star Type"):
    # Encode inputs
    color_encoded = le_color.transform([star_color])[0]
    spectral_encoded = le_spectral.transform([spectral_class])[0]

    # Create feature array
    features = np.array([[temperature, luminosity, radius, absolute_magnitude, star_color, spectral_class]])
    features[:, 4] = color_encoded
    features[:, 5] = spectral_encoded
    features = features.astype(float)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Mapping for display
    star_types = {
        0: "Red Dwarf",
        1: "Brown Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Supergiant",
        5: "Hypergiant"
    }

    st.success(f"Predicted Star Type: **{star_types[prediction]}**")
