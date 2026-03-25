import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_full_signal

st.set_page_config(page_title="FocusGuard", layout="centered")

st.title("FocusGuard: EDA Activity Classifier")
st.write("Classifies Rest vs Aerobic Exercise from EDA signals")

mode = st.selectbox("Demo Mode", ["Upload Real Data", "Simulate"])

# =========================================================
# UPLOAD MODE (REAL DATA)
# =========================================================
if mode == "Upload Real Data":
    file = st.file_uploader("Upload CSV with 'eda' column", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "eda" not in df.columns:
            st.error("CSV must contain 'eda' column")
        else:
            st.subheader("EDA Signal")
            st.line_chart(df["eda"])

            try:
                result = predict_full_signal(df)

                st.subheader("Segment Predictions")

                st.write("REST (0–120 sec):")
                if result["rest_pred"] == 1:
                    st.error(f"Predicted Exercise ({result['rest_conf']*100:.1f}%)")
                else:
                    st.success(f"Rest ({result['rest_conf']*100:.1f}%)")

                st.write("AEROBIC (1560–1740 sec):")
                if result["aero_pred"] == 1:
                    st.success(f"Exercise ({result['aero_conf']*100:.1f}%)")
                else:
                    st.error(f"Predicted Rest ({result['aero_conf']*100:.1f}%)")

                st.caption("Segments: 0–120s = Rest | 1560–1740s = Exercise")

            except Exception as e:
                st.error(str(e))

# =========================================================
# SIMULATION MODE (FOR DEMO)
# =========================================================
elif mode == "Simulate":

    t = np.linspace(0, 1740, 1740 * 4)

    # rest segment
    rest = 0.5 + 0.05*np.sin(t[:480]) + 0.02*np.random.randn(480)

    # middle junk
    mid = 0.5 + 0.02*np.random.randn(len(t) - 480 - 720)

    # exercise segment
    exercise = 0.5 + 0.5*np.sin(5*t[:720]) + 0.1*np.random.randn(720)

    signal = np.concatenate([rest, mid, exercise])
    df = pd.DataFrame({"eda": signal})

    st.subheader("Simulated Full Signal")
    st.line_chart(df["eda"])

    result = predict_full_signal(df)

    st.subheader("Predictions")

    st.write("REST segment:")
    st.success(f"Rest ({result['rest_conf']*100:.1f}%)")

    st.write("AEROBIC segment:")
    st.success(f"Exercise ({result['aero_conf']*100:.1f}%)")

st.markdown("---")
st.caption("Model: Linear Discriminant Analysis (LDA)")
st.caption("Features: Physiological EDA signal features")
