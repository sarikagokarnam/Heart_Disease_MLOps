import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ======================================
# 1. LOAD DATASET
# ======================================

data = pd.read_csv("data/heart_disease.csv")

print("======================================")
print("     HEART DISEASE PREDICTION")
print("======================================")

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ======================================
# 2. CHECK MISSING VALUES
# ======================================

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# ======================================
# 3. SEPARATE FEATURES AND TARGET
# ======================================

X = data.drop("target", axis=1)
y = data["target"]

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ======================================
# 4. TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ======================================
# 5. MLFLOW SETUP
# ======================================

mlflow.set_experiment("Heart Disease Prediction")

mlflow.start_run()

# Log parameters
mlflow.log_param("model", "Random Forest")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
mlflow.log_param("test_size", 0.20)


# ======================================
# 6. BUILD MODEL
# ======================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\n========== MODEL ==========")
print(model)


# ======================================
# 7. TRAIN MODEL
# ======================================

print("\n========== MODEL TRAINING ==========")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ======================================
# 8. MAKE PREDICTIONS
# ======================================

y_pred = model.predict(X_test)

print("\n========== SAMPLE PREDICTIONS ==========")

print("Actual values:")
print(y_test.values[:10])

print("\nPredicted values:")
print(y_pred[:10])


# ======================================
# 9. MODEL EVALUATION
# ======================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ======================================
# 10. LOG METRICS TO MLFLOW
# ======================================

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)


# ======================================
# 11. CLASSIFICATION REPORT
# ======================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)


# ======================================
# 12. CONFUSION MATRIX
# ======================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ======================================
# 13. CREATE CONFUSION MATRIX IMAGE
# ======================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Disease Prediction - Confusion Matrix")

plt.tight_layout()

plt.savefig("reports/confusion_matrix.png")

plt.close()

print("\nConfusion matrix saved:")
print("reports/confusion_matrix.png")


# ======================================
# 14. LOG CONFUSION MATRIX TO MLFLOW
# ======================================

mlflow.log_artifact(
    "reports/confusion_matrix.png"
)


# ======================================
# 15. SAVE MODEL USING JOBLIB
# ======================================

joblib.dump(
    model,
    "models/heart_disease_model.pkl"
)

print("\n========== MODEL SAVED ==========")

print("Saved model:")
print("models/heart_disease_model.pkl")


# ======================================
# 16. LOG MODEL TO MLFLOW
# ======================================

print("\n========== LOGGING MODEL TO MLFLOW ==========")

mlflow.sklearn.log_model(
    model,
    name="heart_disease_model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)

print("Model logged to MLflow successfully!")


# ======================================
# 17. END MLFLOW RUN
# ======================================

mlflow.end_run()


# ======================================
# 18. FINAL MESSAGE
# ======================================

print("\n========== MLFLOW TRACKING ==========")

print("Experiment: Heart Disease Prediction")
print("Parameters logged successfully")
print("Metrics logged successfully")
print("Confusion matrix logged successfully")
print("Model logged successfully")

print("\n======================================")
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("======================================")