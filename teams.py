import streamlit as st
import pickle
import os
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Shopping Prediction",
    page_icon="🛒",
    layout="centered"
)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "student_scaler.pkl")

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found in repository")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error("❌ Scaler file not found in repository")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()

# ---------------- UI ----------------
st.title("🛒 Shopping App")
st.write("Predict whether a person is likely to **purchase**.")

age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
salary = st.number_input(
    "Enter Estimated Salary",
    min_value=1_000,
    max_value=200_000,
    value=50_000,
    step=1000
)

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Person is likely to PURCHASE")
    else:
        st.warning("❌ Person is NOT likely to purchase")
