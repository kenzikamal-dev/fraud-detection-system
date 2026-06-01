import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detection SaaS", layout="wide")

st.title("🚨 Fraud Detection System (SaaS Dashboard)")
st.write("Real-time AI-powered fraud detection using Machine Learning")

# ----------------------------
# SIDEBAR (SAAS INFO PANEL)
# ----------------------------
st.sidebar.title("📊 System Info")

st.sidebar.markdown("""
**Machine Learning Model**  
Fraud Detection System  

**Model Type**  
Random Forest Classifier  

**Explainable AI**  
Enabled (Feature Importance)  

**Dataset**  
Credit Card Transactions  
""")

# ----------------------------
# LOAD MODEL
# ----------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# LOAD DATA FOR METRICS
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
# MODEL EXPLAINABILITY
# ----------------------------
st.markdown("---")
st.subheader("📊 Model Explainability (Feature Importance)")

importances = model.feature_importances_
feature_names = X.columns

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.barh(feature_names[:15], importances[:15])
ax1.set_title("Top Feature Importance")
st.pyplot(fig1)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
st.subheader("📊 Model Performance (Confusion Matrix)")

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")

st.pyplot(fig2)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Fraud Detection SaaS | Powered by Random Forest + Streamlit")