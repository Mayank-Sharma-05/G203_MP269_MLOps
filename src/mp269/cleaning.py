"""Stage 2: Clean the original training dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILE = Path("data/raw/train.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "train_clean.csv"

TARGET = "gb_per_day"

FEATURES = [
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
    "phone_tier",
    "circle",
]


def clean_training_data(input_file=RAW_FILE):
    """Clean training data without modifying the original CSV."""

    df = pd.read_csv(input_file)
    original_rows = len(df)

    # 1. Remove exact duplicate rows.
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # 2. Replace sentinel readings with missing values.
    df.loc[df["video_minutes"] == -999, "video_minutes"] = np.nan

    # 3. Correct stuck sensor readings.
    # Dates are ISO strings, so lexical comparison is chronological.
    dates = df["date"].astype(str)

    stuck_period = (
        (dates >= "2026-04-12")
        & (dates <= "2026-04-25")
        & (df["wifi_hours"] == 4.7)
    )

    stuck_values_removed = int(stuck_period.sum())
    df.loc[stuck_period, "wifi_hours"] = np.nan

    # 4. Detect and correct target values approximately 10x too large.
    sorted_values = (
        df[TARGET].dropna().sort_values(ascending=False).to_numpy()
    )

    max_check = max(5, int(len(sorted_values) * 0.05))

    ratios = (
        sorted_values[:max_check]
        / np.maximum(sorted_values[1:max_check + 1], 1e-9)
    )

    max_gap_index = int(np.argmax(ratios))
    max_gap_ratio = float(ratios[max_gap_index])

    target_typos_fixed = 0

    if max_gap_ratio >= 2.0:
        upper_normal = sorted_values[max_gap_index + 1]
        lower_typo = sorted_values[max_gap_index]
        threshold = (upper_normal + lower_typo) / 2.0

        typo_mask = df[TARGET] > threshold
        target_typos_fixed = int(typo_mask.sum())

        df.loc[typo_mask, TARGET] = df.loc[typo_mask, TARGET] / 10.0

    # 5. Impute numerical features using training medians.
    numeric_features = [
        "age",
        "screen_hours",
        "apps_installed",
        "plan_gb_limit",
        "video_minutes",
        "wifi_hours",
    ]

    for column in numeric_features:
        median_value = df[column].median()
        df.loc[:, column] = df[column].fillna(median_value)

    # 6. Impute categorical features using their modes.
    categorical_features = ["phone_tier", "circle"]

    for column in categorical_features:
        mode = df[column].mode(dropna=True)

        if not mode.empty:
            df.loc[:, column] = df[column].fillna(mode.iloc[0])

    # 7. Exclude identifiers, dates and the leakage column.
    cleaned_df = df[FEATURES + [TARGET]].copy()

    # 8. Validate the cleaned dataset.
    if cleaned_df.isnull().any().any():
        raise ValueError("Missing values remain after cleaning.")

    numeric_df = cleaned_df.select_dtypes(include=np.number)

    if not np.isfinite(numeric_df).all().all():
        raise ValueError("Non-finite numerical values found.")

    if (cleaned_df[TARGET] < 0).any():
        raise ValueError("Negative target values found.")

    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned_df),
        "duplicates_removed": duplicates_removed,
        "stuck_sensor_values_replaced": stuck_values_removed,
        "target_typos_fixed": target_typos_fixed,
        "target_gap_ratio": round(max_gap_ratio, 3),
    }

    return cleaned_df, report


def main():
    cleaned_df, report = clean_training_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("STAGE 2 DATA CLEANING REPORT")
    print("=" * 60)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Saved to:", OUTPUT_FILE)

    print("\nRemaining missing values:")
    print(cleaned_df.isnull().sum())

    print("\nTarget summary:")
    print(cleaned_df[TARGET].describe())


if __name__ == "__main__":
    main()
