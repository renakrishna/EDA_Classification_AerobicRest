import numpy as np

mode = st.selectbox("Demo Mode", ["Upload File", "Simulate"])

if mode == "Simulate":
    t = np.linspace(0, 10, 500)
    signal = 0.5 + 0.1*np.sin(t)  # rest
    if st.button("Simulate Exercise"):
        signal = 0.5 + 0.5*np.sin(5*t)

    df = pd.DataFrame({"eda": signal})
    st.line_chart(df['eda'])

    pred = predict_from_df(df)
    st.write("Prediction:", "Exercise" if pred==1 else "Rest")
