import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Fraud Detection System")

st.title("🚨 Fraud Detection System")

st.write("Simple ML-powered fraud detection demo")

# Load model safely (update path if needed)
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None
    st.warning("Model not found. Check models folder.")

amount = st.number_input("Transaction Amount", 0.0)

if st.button("Predict"):
    if model:
        prediction = model.predict([[amount]])
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.error("Model not loaded")