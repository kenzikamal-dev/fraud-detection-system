import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("🚨 Fraud Detection System")
st.write("Simple ML-powered fraud detection demo using Random Forest")

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "models/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    model = None
    st.error("❌ Model not found. Check models/model.pkl")
    st.stop()

# ---------------------------
# Inputs
# ---------------------------
st.subheader("Transaction Input")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
time_value = st.number_input("Transaction Time", min_value=0.0, value=0.0)

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Fraud"):

    if model:

        # ---------------------------------------
        # IMPORTANT:
        # Model expects 30 features:
        # Time + V1–V28 + Amount
        # ---------------------------------------

        features = np.zeros(30)

        # Time feature (index 0)
        features[0] = time_value

        # Amount feature (last index)
        features[-1] = amount

        # Reshape for sklearn
        input_data = features.reshape(1, -1)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Output
        if prediction == 1:
            st.error("🚨 Fraud Detected")
            st.warning("High-risk transaction flagged by AI model")
        else:
            st.success("✅ Legit Transaction")
            st.info("Transaction appears safe")

    else:
        st.error("Model not loaded")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Fraud Detection System | Streamlit + Scikit-learn")