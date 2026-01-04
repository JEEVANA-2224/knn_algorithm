import streamlit as st
import joblib
import numpy as np

# Load trained model
AI_model = joblib.load("Social_Network_Streamlit.pkl")

# App title
st.title("🛒 Purchase Prediction App")

st.write("Predict whether a person will purchase or not")

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
salary = st.number_input("Enter Salary", min_value=10000, max_value=200000, value=50000)

# Predict button
if st.button("Predict"):
    prediction = AI_model.predict([[age, salary]])

    if prediction[0] == 0:
        st.error("❌ PERSON DID NOT PURCHASE")
    else:
        st.success("✅ PERSON PURCHASED")


