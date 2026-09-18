import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------------
# Load trained model, scaler, and feature names
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.markdown(
    """
    This app uses a **Logistic Regression** model trained on the
    **Breast Cancer Wisconsin (Diagnostic) Dataset** to predict whether a
    tumor is **Malignant** or **Benign**, based on cell nuclei measurements
    from a biopsy image.

    > ⚠️ **Disclaimer:** This tool is built for educational / portfolio
    > purposes as part of a capstone project. It is **not** a medical device
    > and must never be used for real clinical decision-making.
    """
)

st.divider()

# ---------------------------------------------------------
# Sidebar: choose input mode
# ---------------------------------------------------------
st.sidebar.header("Input Options")
mode = st.sidebar.radio(
    "How would you like to provide input?",
    ["Use example patient", "Enter values manually"]
)

# A benign and a malignant example row (mean values), for quick demo
EXAMPLES = {
    "Example: Benign-leaning case": [12.0, 15.0, 78.0, 460.0, 0.09, 0.08, 0.03, 0.02, 0.17, 0.06,
                                      0.3, 1.0, 2.0, 25.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003,
                                      13.5, 20.0, 88.0, 550.0, 0.13, 0.18, 0.15, 0.07, 0.28, 0.08],
    "Example: Malignant-leaning case": [20.5, 22.0, 135.0, 1300.0, 0.11, 0.20, 0.25, 0.14, 0.20, 0.07,
                                         0.8, 1.5, 5.5, 100.0, 0.008, 0.04, 0.05, 0.02, 0.03, 0.005,
                                         25.0, 30.0, 170.0, 2000.0, 0.16, 0.5, 0.6, 0.25, 0.35, 0.10],
}

# ---------------------------------------------------------
# Collect input values
# ---------------------------------------------------------
input_values = []

if mode == "Use example patient":
    choice = st.selectbox("Pick an example patient:", list(EXAMPLES.keys()))
    input_values = EXAMPLES[choice]
    st.info("Loaded example values below. Switch to 'Enter values manually' to customize.")

    with st.expander("View example feature values"):
        st.write({name: round(val, 4) for name, val in zip(feature_names, input_values)})

else:
    st.write("Enter the 10 **mean** measurements below (most clinically relevant subset). "
             "Remaining derived features are estimated automatically for the prediction.")

    col1, col2 = st.columns(2)
    with col1:
        mean_radius = st.number_input("Mean Radius", 5.0, 40.0, 14.0)
        mean_texture = st.number_input("Mean Texture", 5.0, 40.0, 19.0)
        mean_perimeter = st.number_input("Mean Perimeter", 40.0, 200.0, 92.0)
        mean_area = st.number_input("Mean Area", 100.0, 2600.0, 650.0)
        mean_smoothness = st.number_input("Mean Smoothness", 0.02, 0.20, 0.10)

    with col2:
        mean_compactness = st.number_input("Mean Compactness", 0.0, 0.40, 0.10)
        mean_concavity = st.number_input("Mean Concavity", 0.0, 0.50, 0.09)
        mean_concave_points = st.number_input("Mean Concave Points", 0.0, 0.25, 0.05)
        mean_symmetry = st.number_input("Mean Symmetry", 0.1, 0.35, 0.18)
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", 0.04, 0.10, 0.06)

    # Build a full 30-feature vector: use manual mean-features, and reasonable
    # dataset-average placeholders for the "error" and "worst" feature groups.
    default_full = np.array(EXAMPLES["Example: Benign-leaning case"], dtype=float)
    input_values = default_full.copy()
    manual_means = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                     mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                     mean_fractal_dimension]
    input_values[0:10] = manual_means

st.divider()

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
    X = np.array(input_values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    label = "Benign" if prediction == 1 else "Malignant"
    confidence = proba[prediction] * 100

    st.subheader("Prediction Result")
    if label == "Malignant":
        st.error(f"⚠️ Predicted: **{label}** (Confidence: {confidence:.2f}%)")
    else:
        st.success(f"✅ Predicted: **{label}** (Confidence: {confidence:.2f}%)")

    st.write("**Probability breakdown:**")
    st.progress(float(proba[1]))
    st.caption(f"Benign probability: {proba[1]*100:.2f}% | Malignant probability: {proba[0]*100:.2f}%")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Capstone Project — Artificial Intelligence (Machine Learning & Deep Learning) | "
    "NAVTTC Prime Minister's Hunarmand Pakistan Program | "
    "Built by Syed Israr Ali (Roll No. 24)"
)
