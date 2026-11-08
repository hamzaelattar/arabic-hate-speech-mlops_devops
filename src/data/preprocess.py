from pathlib import Path
import argparse
import re

import emoji
import pandas as pd

from src.pipeline.paths import RAW_DIR, PROCESSED_DIR, project_path


# Automatically locate the root when this file is at src/data/preprocess.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ar_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LABEL_COLS = [
    "target", "sentiment", "directness", "group", "annotator_sentiment"
]
SENTIMENT_LABELS = [
    "abusive", "hateful", "offensive", "normal", "disrespectful", "fearful"
]


def clean_arabic(text: str) -> str:
    """Arabic normalization adapted from the notebook's clean_arabic (cell 45)."""
    text = str(text)

    # 1. Remove real URLs and the dataset's @url placeholder.
    text = re.sub(
        r"https?://\S+|www\.\S+|@url\b", " ", text,
        flags=re.IGNORECASE,
    )

    # 2. Preserve anonymized user mentions as a uniform token.
    text = re.sub(r"@user\b", " USER ", text, flags=re.IGNORECASE)

    # 3. Remove emojis (same emoji library used in the notebook).
    text = emoji.replace_emoji(text, replace=" ")

    # 4. Remove the hashtag marker, not the hashtag word.
    text = re.sub(r"(?<!\w)#(?=\w)", "", text)

    # 5. Remove Arabic diacritics.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # 6. Remove Tatweel.
    text = text.replace("ـ", "")

    # 7. Normalize Alef variants.
    text = re.sub(r"[أإآٱ]", "ا", text)

    # 8. Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_arabic(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    mode: str = "train",
) -> pd.DataFrame:
    """Clean Arabic tweets, remove conflicting labels and save audit files."""
    input_path = project_path(input_path, RAW_DIR, file=True)
    output_dir = project_path(output_dir, PROCESSED_DIR)
    if mode not in {"train", "evaluate"}:
        raise ValueError("mode must be train or evaluate")

    data = pd.read_csv(input_path, encoding="utf-8-sig")
    if "tweet" not in data.columns:
        raise ValueError("Missing required column: tweet")
    if mode == "evaluate":
        data["tweet_clean"] = data["tweet"].fillna("").apply(clean_arabic)
        data = data.loc[data["tweet_clean"].ne("")].copy()
        if "target" in data.columns:
            data["target"] = data["target"].replace({"disability": "other"})
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_dir / "ar_ready.csv", index=False, encoding="utf-8-sig")
        print(f"External rows ready: {len(data)}")
        return data

    required_cols = {"tweet", *LABEL_COLS}
    missing = sorted(required_cols - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_size = len(data)

    # Notebook cell 13: a sentiment can have multiple labels, separated by '_'.
    split_labels = data["sentiment"].fillna("").astype(str).str.split("_")
    for label in SENTIMENT_LABELS:
        data[f"sentiment_{label}"] = split_labels.apply(
            lambda labels: int(label in labels)
        )

    # Notebook cell 19: merge the single rare Arabic target class.
    data["target"] = data["target"].replace({"disability": "other"})

    # Light normalization for initial duplicate/conflict detection only.
    data["_normalized"] = (
        data["tweet"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Remove ALL rows for any original text whose annotations disagree.
    conflicting_groups = (
        data.groupby("_normalized")[LABEL_COLS]
        .nunique(dropna=False).gt(1).any(axis=1)
    )
    conflict_keys = conflicting_groups[conflicting_groups].index
    conflicts = data[data["_normalized"].isin(conflict_keys)].copy()
    data = data.loc[~data["_normalized"].isin(conflict_keys)].copy()
    after_conflicts = len(data)

    # Remove remaining duplicate original texts.
    data = data.drop_duplicates(subset="_normalized", keep="first")
    duplicates_removed = after_conflicts - len(data)
    data = data.drop(columns=["_normalized"]).reset_index(drop=True)
    conflicts = conflicts.drop(columns=["_normalized"])

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "ar_clean.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(
        output_dir / "ar_conflicts.csv", index=False, encoding="utf-8-sig"
    )

    # Notebook cells 45-46: clean text without changing its original column.
    data["tweet_clean"] = data["tweet"].apply(clean_arabic)
    data.to_csv(output_dir / "ar_final.csv", index=False, encoding="utf-8-sig")

    # Remove empty texts; they should never be used for model training.
    n_empty = int(data["tweet_clean"].eq("").sum())
    data = data.loc[data["tweet_clean"].ne("")].copy()

    # NEW: Detect disagreements AFTER Arabic normalization, across LABEL_COLS.
    # Two different raw tweets may collapse to the same cleaned text.
    clean_conflicting_groups = (
        data.groupby("tweet_clean")[LABEL_COLS]
        .nunique(dropna=False).gt(1).any(axis=1)
    )
    clean_conflict_keys = clean_conflicting_groups[
        clean_conflicting_groups
    ].index
    n_conflicting_clean_groups = len(clean_conflict_keys)

    # Save ALL examples from conflicting cleaned-text groups for auditing.
    clean_conflicts = data.loc[
        data["tweet_clean"].isin(clean_conflict_keys)
    ].copy()
    clean_conflicts.to_csv(
        output_dir / "ar_clean_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # NEW: Remove ALL conflicting groups, not just the duplicate copies.
    data = data.loc[
        ~data["tweet_clean"].isin(clean_conflict_keys)
    ].copy()

    # Only deduplicate cleaned texts with consistent annotations.
    n_duplicates_after_clean = int(data["tweet_clean"].duplicated().sum())
    data_ready = data.drop_duplicates(
        subset="tweet_clean", keep="first"
    ).reset_index(drop=True)
    data_ready.to_csv(
        output_dir / "ar_ready.csv", index=False, encoding="utf-8-sig"
    )

    print("=== Arabic preprocessing (conflicts removed before AND after cleaning) ===")
    print(f"Raw tweets:                       {initial_size}")
    print(f"Initial conflicting rows removed: {len(conflicts)}")
    print(f"Initial duplicates removed:       {duplicates_removed}")
    print(f"Empty cleaned tweets removed:     {n_empty}")
    print(f"Conflicting cleaned groups:       {n_conflicting_clean_groups}")
    print(f"Conflicting cleaned rows removed: {len(clean_conflicts)}")
    print(f"Post-clean duplicates removed:    {n_duplicates_after_clean}")
    print(f"Final tweets:                     {len(data_ready)}")
    print("\nTarget distribution:")
    print(data_ready["target"].value_counts(dropna=False).to_string())
    print(f"\nProcessed files saved to: {output_dir}")

    return data_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    args = parser.parse_args()
    preprocess_arabic(args.input, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
