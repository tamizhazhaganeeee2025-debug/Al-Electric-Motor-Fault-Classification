"""
Streamlit dashboard for AI-Based Electric Motor Fault Classification.
Run: streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "motor_fault_demo.csv"
MODEL_PATH = ROOT / "motor_fault_model.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"
FEATURES = [
    "vibration_rms", "vibration_std", "peak_amplitude", "kurtosis",
    "dominant_frequency_hz", "temperature_c", "current_a", "acoustic_rms"
]
LABEL_NAMES = {
    "Normal": "Normal",
    "Bearing_Fault": "Bearing fault",
    "Imbalance_Fault": "Imbalance fault",
    "Misalignment_Fault": "Misalignment fault"
}

st.set_page_config(page_title="Motor Fault Classification", page_icon="⚙️", layout="wide")
st.title("⚙️ AI-Based Electric Motor Fault Classification")
st.caption("Educational machine-learning prototype | Not a certified safety or maintenance system")
st.warning("The bundled dataset is SIMULATED demo data. Predictions and metrics are for demonstrating the software workflow only; they are not evidence of real-world diagnostic performance.")

@st.cache_resource
def load_or_train():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["classes"]
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"].astype(str)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model, sorted(y.unique())

model, classes = load_or_train()
df = pd.read_csv(DATA_PATH)

tab1, tab2, tab3 = st.tabs(["Predict condition", "Dataset explorer", "Model evaluation"])
with tab1:
    st.subheader("Enter sample measurements")
    st.write("Use measurements from a suitable dataset or approved laboratory setup. Units and sensor calibration must match your source.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vibration_rms = st.number_input("Vibration RMS", min_value=0.0, value=0.60, step=0.05)
        vibration_std = st.number_input("Vibration standard deviation", min_value=0.0, value=0.25, step=0.05)
    with c2:
        peak_amplitude = st.number_input("Peak amplitude", min_value=0.0, value=1.80, step=0.1)
        kurtosis = st.number_input("Kurtosis", min_value=0.0, value=3.0, step=0.1)
    with c3:
        dominant_frequency_hz = st.number_input("Dominant frequency (Hz)", min_value=0.0, value=25.0, step=1.0)
        temperature_c = st.number_input("Temperature (°C)", value=42.0, step=1.0)
    with c4:
        current_a = st.number_input("Current (A)", min_value=0.0, value=2.2, step=0.1)
        acoustic_rms = st.number_input("Acoustic RMS", min_value=0.0, value=0.12, step=0.01)
    if st.button("Classify motor condition", type="primary"):
        sample = pd.DataFrame([{
            "vibration_rms": vibration_rms, "vibration_std": vibration_std,
            "peak_amplitude": peak_amplitude, "kurtosis": kurtosis,
            "dominant_frequency_hz": dominant_frequency_hz, "temperature_c": temperature_c,
            "current_a": current_a, "acoustic_rms": acoustic_rms
        }], columns=FEATURES)
        prediction = model.predict(sample)[0]
        st.success(f"Predicted class: **{LABEL_NAMES.get(prediction, prediction)}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            prob_df = pd.DataFrame({"Class": [LABEL_NAMES.get(c, c) for c in model.classes_], "Model score": probabilities})
            st.bar_chart(prob_df.set_index("Class"))
        st.info("This prediction is based on the training data. Verify any suspected fault using qualified personnel and approved procedures.")

with tab2:
    st.subheader("Dataset preview")
    st.write(f"Rows: {len(df)} | Features: {len(FEATURES)} | Classes: {df['label'].nunique()}")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Class distribution")
    st.bar_chart(df["label"].value_counts())
    selected_feature = st.selectbox("Choose feature to plot", FEATURES)
    fig, ax = plt.subplots()
    for label, group in df.groupby("label"):
        ax.hist(group[selected_feature], bins=25, alpha=0.45, label=LABEL_NAMES.get(label, label))
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Hold-out evaluation")
    X = df[FEATURES]
    y = df["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    eval_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
    ])
    eval_model.fit(X_train, y_train)
    pred = eval_model.predict(X_test)
    st.metric("Demo hold-out accuracy", f"{accuracy_score(y_test, pred):.1%}")
    st.caption("This metric is calculated from simulated demo data; do not present it as measured real-world performance.")
    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=labels), display_labels=[LABEL_NAMES.get(x, x) for x in labels]).plot(ax=ax, xticks_rotation=30, colorbar=False)
    st.pyplot(fig)
    plt.close(fig)
    st.text("Classification report")
    st.code(classification_report(y_test, pred, zero_division=0))
