import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from mp269.train import (
    RAW_TRAIN,
    CLEAN_TRAIN,
    FEATURES,
    TARGET,
    build_pipeline,
    calculate_sha256,
)


def test_original_train_sha256():
    """Check SHA-256 against the original raw training CSV."""
    expected = hashlib.sha256(RAW_TRAIN.read_bytes()).hexdigest()

    actual = calculate_sha256(RAW_TRAIN)

    assert actual == expected
    assert len(actual) == 64


def test_model_pipeline_training_and_prediction():
    """Check that the ML pipeline trains and returns valid predictions."""
    data = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45],
        "screen_hours": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "apps_installed": [20, 30, 40, 50, 60, 70],
        "plan_gb_limit": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "video_minutes": [30, 60, 90, 120, 150, 180],
        "wifi_hours": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "phone_tier": ["mid", "mid", "flagship", "mid", "basic", "flagship"],
        "circle": ["Urban", "Rural", "Urban", "Semi-urban", "Rural", "Urban"],
    })

    targets = pd.Series([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    model = build_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42)
    )

    model.fit(data[FEATURES], targets)
    predictions = model.predict(data[FEATURES])

    assert len(predictions) == len(targets)
    assert np.isfinite(predictions).all()


def test_cleaned_training_dataset():
    """Check the generated cleaned dataset."""
    assert Path(CLEAN_TRAIN).exists()

    df = pd.read_csv(CLEAN_TRAIN)

    assert len(df) == 1000
    assert TARGET in df.columns
    assert all(feature in df.columns for feature in FEATURES)
    assert df.isnull().sum().sum() == 0
    assert (df[TARGET] >= 0).all()

    # Leakage and identifier columns must not be model features.
    assert "gb_from_monthly_bill" not in df.columns
    assert "record_id" not in df.columns
    assert "date" not in df.columns
