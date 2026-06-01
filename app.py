import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detection SaaS", layout="wide")

st.title("🚨 Fraud Detection System (SaaS Dashboard)")
st.write("Real-time AI-powered fraud detection using Random Forest")

# ----------------------------
# SIDEBAR (SYSTEM INFO)
# ----------------------------
st.sidebar.title("📊 System Info")

st.sidebar.markdown("""
**Machine Learning Model**  
Random Forest Classifier  

**Task**  
Fraud Detection  

**Explainable AI**  
Feature Importance Enabled  
""")

# ----------------------------
# LOAD MODEL
# ----------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# LOAD DATA (FOR METRICS)
# ----------------------------
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ----------------------------
# INPUT SECTION
# ----------------------------
st.subheader("💳 Transaction Input")

time = st.number_input("Time", 0.0)

v_features = []
cols = st.columns(4)

for i in range(28):
    with cols[i % 4]:
        v = st.number_input(f"V{i+1}", 0.0)
        v_features.append(v)

amount = st.number_input("Amount", 0.0)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Fraud"):

    features = [time] + v_features + [amount]
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legit Transaction")

# ----------------------------
# FEATURE IMPORTANCE (FIXED)
# ----------------------------
st.markdown("---")
st.subheader("📊 Model Explainability (Feature Importance)")

importances = model.feature_importances_
feature_names = np.array(X.columns)

# safe sorting
idx = np.argsort(importances)[::-1][:15]

top_features = feature_names[idx]
top_importances = importances[idx]

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(top_features, top_importances)
ax.invert_yaxis()
ax.set_title("Top Feature Importance")

st.pyplot(fig)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
st.subheader("📊 Model Performance (Confusion Matrix)")

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()

ax2.imshow(cm, cmap="Blues")
ax2.set_title("Confusion Matrix")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig2)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Fraud Detection SaaS | Streamlit + RandomForest")