
from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline.paths import PROCESSED_DIR, project_path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ar_ready.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def split_data(input_path: str | Path = INPUT_PATH, output_dir: str | Path = OUTPUT_DIR):
    input_path = project_path(input_path, PROCESSED_DIR, file=True)
    output_dir = project_path(output_dir, PROCESSED_DIR)
    # 1. Load the cleaned dataset
    data = pd.read_csv(input_path)
    missing = {"tweet_clean", "target"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["target"].isna().any():
        raise ValueError("Training target contains missing labels")

    print(f"Total samples: {len(data)}")

    # 2. Split into 70% train and 30% temporary
    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        random_state=42,
        stratify=data["target"]
    )

    # 3. Split temporary data into 15% validation and 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["target"]
    )

    # 4. Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 5. Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save datasets
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(output_dir / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    # 7. Display results
    print("\nDataset split completed!")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    # 8. Check class distribution
    for name, df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"\n{name} target distribution:")
        print(df["target"].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    split_data(args.input, args.output_dir)
