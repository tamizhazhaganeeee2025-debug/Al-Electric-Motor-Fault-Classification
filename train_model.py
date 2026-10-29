"""
Train and evaluate an electric motor fault classifier.
Run: python train_model.py
IMPORTANT: bundled CSV is simulated teaching/demo data, not real sensor measurements.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "motor_fault_demo.csv"
MODEL_PATH = ROOT / "motor_fault_model.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"
FEATURES = [
    "vibration_rms", "vibration_std", "peak_amplitude", "kurtosis",
    "dominant_frequency_hz", "temperature_c", "current_a", "acoustic_rms"
]

def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=["label"])
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=sorted(y.unique()))
    results = {
        "accuracy": float(accuracy),
        "classes": sorted(y.unique()),
        "features": FEATURES,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": sorted(y.unique()),
        "dataset_rows": int(len(df)),
        "test_rows": int(len(y_test)),
        "dataset_note": "Bundled data is simulated for software demonstration only."
    }
    joblib.dump({"model": model, "features": FEATURES, "classes": sorted(y.unique())}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Hold-out accuracy on simulated demo data: {accuracy:.3f}")
    print(json.dumps(classification_report(y_test, predictions, zero_division=0), indent=2))
    print("Reminder: results on simulated data do not establish real-world motor fault accuracy.")

if __name__ == "__main__":
    main()
