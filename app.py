import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_from_df

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FocusGuard", layout="centered")

# ---------------- HEADER ----------------
st.title("FocusGuard: EDA Activity Classifier")
st.write("Classify Rest vs Exercise using only EDA signals")

# ---------------- MODE SELECT ----------------
mode = st.selectbox("Demo Mode", ["Upload File", "Simulate"])

# =========================================================
# ---------------- UPLOAD MODE ----------------
# =========================================================
if mode == "Upload File":
    uploaded_file = st.file_uploader("Upload EDA CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'eda' not in df.columns:
                st.error("CSV must contain a column named 'eda'")
            else:
                st.subheader("EDA Signal")
                st.line_chart(df['eda'])

                pred = predict_from_df(df)

                st.subheader("Prediction")
                if pred == 1:
                    st.success("Exercise")
                else:
                    st.info("Rest")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# =========================================================
# ---------------- SIMULATION MODE ----------------
# =========================================================
elif mode == "Simulate":

    st.subheader("Generate Synthetic EDA Signal")

    sim_type = st.selectbox("Signal Type", ["Rest", "Exercise"])

    t = np.linspace(0, 10, 500)

    if sim_type == "Rest":
        signal = 0.5 + 0.05*np.sin(t) + 0.02*np.random.randn(len(t))
    else:
        signal = 0.5 + 0.4*np.sin(5*t) + 0.1*np.random.randn(len(t))

    df = pd.DataFrame({"eda": signal})

    st.subheader("Simulated EDA Signal")
    st.line_chart(df['eda'])

    pred = predict_from_df(df)

    st.subheader("Prediction")
    if pred == 1:
        st.success("Exercise")
    else:
        st.info("Rest")

# =========================================================
# ---------------- FOOTER (EXPLAINS YOUR PROJECT) ----------------
# =========================================================
st.markdown("---")
st.caption("Model: Extra Trees Classifier")
st.caption("Features: Statistical + signal-based EDA features")
