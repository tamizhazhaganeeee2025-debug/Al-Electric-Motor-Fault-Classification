# AI-Based Electric Motor Fault Classification

## Overview
This educational EEE capstone demonstrates a machine-learning workflow for classifying motor condition labels from vibration, temperature, current and acoustic features.

**Important data note:** `data/motor_fault_demo.csv` contains simulated data generated only to let you run and demonstrate the software. It is NOT collected from a real motor and must not be described as real experimental evidence. Replace it with a suitable public, labelled dataset or approved supervised laboratory measurements before making real-world claims.

## Example classes
- Normal
- Bearing_Fault
- Imbalance_Fault
- Misalignment_Fault

These example labels are illustrative. Use only labels supported by your selected dataset.

## Features
- Data preview and class distribution
- Random Forest classifier
- Missing-value imputation
- Hold-out test evaluation, classification report and confusion matrix
- Interactive prediction form
- Feature histogram plots

## Requirements
Python 3.10 or newer recommended.

## Run in Google Colab or on a computer
1. Extract the ZIP file.
2. Open a terminal in the project folder.
3. Install packages:
   `pip install -r requirements.txt`
4. Train and save the model:
   `python train_model.py`
5. Launch the dashboard:
   `streamlit run app.py`
6. Open the local Streamlit URL shown in the terminal.

For Google Colab, upload the folder or files to Colab. A Streamlit tunnel may be required to view the dashboard; for a simpler demo, run the notebook version or run locally.

## Dataset schema
The CSV must contain these numeric feature columns and one label column:
`vibration_rms, vibration_std, peak_amplitude, kurtosis, dominant_frequency_hz, temperature_c, current_a, acoustic_rms, label`

Keep units and sensor sampling/calibration details documented. Do not combine datasets with incompatible measurement units without appropriate processing.

## Methodology
1. Load and inspect labelled measurements.
2. Check missing values, duplicates, outliers, and class balance.
3. Split data into training and test sets using stratification.
4. Impute missing feature values using training-set medians within a pipeline.
5. Train a Random Forest classifier.
6. Evaluate using accuracy, precision, recall, F1-score and confusion matrix.
7. Use the Streamlit dashboard to explore samples and demonstrate predictions.

## Limitations and safety
- Simulated demo metrics are not real-world performance results.
- A model can be unreliable if the training data is small, imbalanced, mislabeled or from a different motor/sensor setup.
- Do not intentionally create faults on a running motor. Use public data or supervised, guarded laboratory experiments.
- Do not use this educational prototype to make safety-critical maintenance decisions.
