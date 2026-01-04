import streamlit as st
import numpy as np
import joblib
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Social Network Ads Prediction",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Social Network Purchase Prediction")
st.write("Predict whether a user will **purchase a product** using KNN.")

# ---------------- Load Model & Scaler ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "student_final_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "student_scaler.pkl"))

# ---------------- User Inputs ----------------
age = st.number_input("Age", min_value=18, max_value=100, value=47)
salary = st.number_input("Estimated Salary", min_value=1000, max_value=200000, value=25000)

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)  # 🔥 VERY IMPORTANT
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ User will PURCHASE the product")
    else:
        st.error("❌ User will NOT purchase the product")
