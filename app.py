import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to get a price prediction.")

sqft = st.number_input("House Size (sq ft)", min_value=500, max_value=10000, value=1500, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000, step=1)

if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms, year_built]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"Estimated House Price: **${prediction:,.2f}**")
