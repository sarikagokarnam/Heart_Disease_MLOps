import pandas as pd

# Column names for the Cleveland Heart Disease dataset
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load original Cleveland dataset
data = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

print("Original dataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

# Remove rows containing missing values
data = data.dropna()

# Convert target:
# 0 = No heart disease
# 1,2,3,4 = Heart disease
data["target"] = data["target"].apply(
    lambda x: 0 if x == 0 else 1
)

# Save cleaned CSV
data.to_csv(
    "data/heart_disease.csv",
    index=False
)

print("\n================================")
print("Dataset conversion completed!")
print("================================")

print("\nFinal dataset shape:")
print(data.shape)

print("\nFinal columns:")
print(data.columns.tolist())

print("\nTarget distribution:")
print(data["target"].value_counts())

print("\nSaved file:")
print("data/heart_disease.csv")