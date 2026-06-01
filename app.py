import streamlit as st
import pickle
import numpy as np

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚨 Fraud Detection System")

st.write("Enter transaction details (30 features model)")

# Inputs
time = st.number_input("Time", 0.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", 0.0)
    v_features.append(v)

amount = st.number_input("Amount", 0.0)

# Predict
if st.button("Predict Fraud"):

    features = [time] + v_features + [amount]
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legit Transaction")