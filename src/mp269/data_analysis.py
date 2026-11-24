"""General exploratory data analysis (EDA) utilities for MP269."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_params(params_path: str | Path = "params.yaml") -> Dict[str, Any]:
    """Load project configuration parameters from params.yaml."""
    path = Path(params_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_datasets(
    params: Dict[str, Any], base_dir: str | Path = "."
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw train, later, and holdout feature datasets without modifying them."""
    root = Path(base_dir)
    raw_dir = root / params["data"]["raw_dir"]

    train_df = pd.read_csv(raw_dir / params["data"]["train_file"])
    later_df = pd.read_csv(raw_dir / params["data"]["later_file"])
    holdout_df = pd.read_csv(raw_dir / params["data"]["holdout_file"])
    return train_df, later_df, holdout_df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic dataset dimensions, column names, and data types."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages for every column."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100.0).round(2)
    return pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values.astype(int),
            "missing_pct": missing_pct.values,
        }
    )


def get_duplicate_summary(df: pd.DataFrame, id_col: str = "record_id") -> Dict[str, int]:
    """Compute exact duplicate row counts and duplicate ID counts."""
    exact_dups = int(df.duplicated().sum())
    id_dups = int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
    return {
        "total_rows": int(len(df)),
        "unique_rows": int(len(df.drop_duplicates())),
        "exact_duplicate_rows": exact_dups,
        "duplicate_record_ids": id_dups,
    }


def get_unique_value_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return the number of unique values in each column."""
    return {col: int(df[col].nunique(dropna=False)) for col in df.columns}


def get_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats = stats.rename(columns={"50%": "median"})
    stats.insert(0, "column", stats.index)
    return stats.reset_index(drop=True)


def get_categorical_distributions(
    df: pd.DataFrame, exclude_cols: List[str] | None = None
) -> Dict[str, pd.DataFrame]:
    """Compute frequency tables for categorical (non-ID, non-date) columns."""
    excluded = set(exclude_cols or ["record_id", "date"])
    cat_cols = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in excluded
    ]
    distributions: Dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        pcts = (counts / len(df) * 100.0).round(2)
        distributions[col] = pd.DataFrame(
            {
                "category": counts.index.astype(str),
                "count": counts.values.astype(int),
                "percentage": pcts.values,
            }
        )
    return distributions


def get_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Detect statistical outliers in numeric columns using the 1.5 * IQR rule."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        low_count = int((series < lower_bound).sum())
        high_count = int((series > upper_bound).sum())
        rows.append(
            {
                "column": col,
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "low_outliers": low_count,
                "high_outliers": high_count,
                "total_outliers": low_count + high_count,
            }
        )
    return pd.DataFrame(rows)


def get_date_distribution(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """Summarize the date range, unique dates, and monthly record counts."""
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    monthly = parsed.dt.to_period("M").value_counts().sort_index()
    daily_counts = parsed.value_counts()
    return {
        "min_date": str(parsed.min().date()),
        "max_date": str(parsed.max().date()),
        "unique_dates": int(parsed.nunique()),
        "invalid_dates": int(parsed.isna().sum()),
        "min_records_per_day": int(daily_counts.min()),
        "max_records_per_day": int(daily_counts.max()),
        "mean_records_per_day": round(float(daily_counts.mean()), 2),
        "monthly_counts": {str(period): int(cnt) for period, cnt in monthly.items()},
    }


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().round(4)


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Run all general exploratory data analysis checks on a dataframe."""
    return {
        "overview": get_dataset_overview(df),
        "missing_values": get_missing_value_summary(df),
        "duplicates": get_duplicate_summary(df),
        "unique_counts": get_unique_value_counts(df),
        "numerical_stats": get_numerical_statistics(df),
        "categorical_distributions": get_categorical_distributions(df),
        "outliers": get_outlier_summary(df),
        "date_distribution": get_date_distribution(df),
        "correlations": get_correlation_matrix(df),
    }
