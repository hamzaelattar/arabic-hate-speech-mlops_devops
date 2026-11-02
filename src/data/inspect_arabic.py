
import pandas as pd

df = pd.read_csv(
    "data/raw/ar_dataset.csv",
    encoding="utf-8-sig"
)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

if "target" in df.columns:
    print("\nClass distribution:")
    print(df["target"].value_counts(dropna=False))
