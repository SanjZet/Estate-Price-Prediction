import streamlit as st
import joblib
import numpy as np

# Load trained Gradient Boosting model
model = joblib.load("/workspaces/Estate-Price-Prediction/model.ipynbb")

st.title("🏠 Estate Price Prediction")
st.write("Predict house price per unit area using Gradient Boosting.")

# Input fields
transaction_date = st.number_input("Transaction Date (e.g., 2013.25)", 2010.0, 2025.0, 2013.25)
house_age = st.number_input("House Age (years)", 0.0, 50.0, 10.0)
mrt_distance = st.number_input("Distance to Nearest MRT Station (meters)", 0.0, 5000.0, 300.0)
convenience_stores = st.number_input("Number of Convenience Stores", 0, 20, 5)
latitude = st.number_input("Latitude", 0.0, 90.0, 24.98)
longitude = st.number_input("Longitude", 0.0, 180.0, 121.54)

# Predict button
if st.button("Predict Price"):
    input_features = np.array([[transaction_date, house_age, mrt_distance,
                                convenience_stores, latitude, longitude]])
    prediction = model.predict(input_features)
    st.success(f"💰 Predicted Price per Unit Area: {prediction[0]:.2f}")
