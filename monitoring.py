import pandas as pd
import joblib

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset
)

print("======================================")
print("   HEART DISEASE MODEL MONITORING")
print("======================================")

# 1. Load dataset
data = pd.read_csv("data/heart_disease.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# 2. Load trained model
model = joblib.load("models/heart_disease_model.pkl")

print("Model loaded successfully!")

# 3. Prepare features and target
X = data.drop("target", axis=1)
y = data["target"]

# Generate predictions
predictions = model.predict(X)

# 4. Create monitoring data
monitoring_data = X.copy()

monitoring_data["target"] = y
monitoring_data["prediction"] = predictions

print("\nMonitoring data prepared!")

# 5. Create reference and current datasets
reference_data = monitoring_data.iloc[:148].copy()
current_data = monitoring_data.iloc[148:].copy()

print("\nReference data shape:", reference_data.shape)
print("Current data shape:", current_data.shape)

# 6. Create Evidently report
report = Report(
    metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ]
)

# 7. Run report
print("\nGenerating Evidently monitoring report...")

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# 8. Save report
report.save_html("reports/evidently_report.html")

print("\n======================================")
print("EVIDENTLY REPORT GENERATED SUCCESSFULLY")
print("======================================")

print("\nReport saved at:")
print("reports/evidently_report.html")