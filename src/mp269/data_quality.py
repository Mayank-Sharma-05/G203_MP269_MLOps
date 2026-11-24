"""Automated detection of the seven hidden data quality problems for MP269."""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Standard unit conversion factors mentioned in STUDENT_HANDBOOK.md Section 7
STANDARD_UNIT_FACTORS = (10, 12, 60, 100, 1000)


def detect_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """Identify columns present in training data but unavailable at prediction time.

    A column missing from holdout_features.csv (other than the target) that strongly
    tracks the target variable is flagged as a leaking column.
    """
    extra_cols = [
        col
        for col in train_df.columns
        if col not in holdout_df.columns and col != target_col
    ]

    if not extra_cols:
        return {
            "present": False,
            "column": None,
            "evidence": {"extra_columns_not_in_holdout": []},
        }

    # Evaluate correlation with the target on both all rows and trimmed rows
    # (excluding extreme target outliers so 10x typos do not mask high correlation).
    q99 = float(train_df[target_col].quantile(0.99))
    trimmed_df = train_df[train_df[target_col] <= q99]

    best_col: Optional[str] = None
    best_trimmed_corr = -1.0
    details: Dict[str, Any] = {}

    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            full_corr = float(train_df[col].corr(train_df[target_col]))
            trimmed_corr = float(trimmed_df[col].corr(trimmed_df[target_col]))
            mad = float((trimmed_df[col] - trimmed_df[target_col]).abs().mean())
            details[col] = {
                "full_correlation": round(full_corr, 6),
                "trimmed_correlation": round(trimmed_corr, 6),
                "trimmed_mean_abs_diff": round(mad, 6),
            }
            if abs(trimmed_corr) > best_trimmed_corr:
                best_trimmed_corr = abs(trimmed_corr)
                best_col = col
        else:
            details[col] = {"note": "non-numeric column missing from holdout"}
            if best_col is None:
                best_col = col

    return {
        "present": best_col is not None,
        "column": best_col,
        "evidence": {
            "extra_columns_not_in_holdout": extra_cols,
            "column_metrics": details,
        },
    }


def detect_sentinel_value(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, Any]:
    """Detect impossible or repeated placeholder sentinel values in numeric columns."""
    common_sentinels = {-9999, -999, -99, -1, 999, 9999, 99999}
    candidates = []

    for col in numeric_cols:
        series = train_df[col].dropna()
        value_counts = series.value_counts()

        # Check negative values or known sentinel numbers appearing multiple times
        for val, count in value_counts.items():
            if count < 5:
                continue
            is_negative_impossible = val < 0
            is_known_extreme = val in common_sentinels and (
                val < series.quantile(0.01) or val > series.quantile(0.99)
            )
            if is_negative_impossible or is_known_extreme:
                other_vals = series[series != val]
                if len(other_vals) > 0:
                    gap = min(
                        abs(val - float(other_vals.min())),
                        abs(val - float(other_vals.max())),
                    )
                else:
                    gap = 0.0
                candidates.append(
                    {
                        "column": col,
                        "value": int(val) if float(val).is_integer() else float(val),
                        "count": int(count),
                        "count_deduped": int(
                            (train_df.drop_duplicates()[col] == val).sum()
                        ),
                        "gap_from_normal_range": round(float(gap), 3),
                    }
                )

    if not candidates:
        return {
            "present": False,
            "column": None,
            "value": None,
            "count": None,
            "evidence": {"candidates": []},
        }

    # Pick the strongest candidate (largest gap from normal data, then highest count)
    candidates.sort(
        key=lambda item: (item["gap_from_normal_range"], item["count"]), reverse=True
    )
    top = candidates[0]
    return {
        "present": True,
        "column": top["column"],
        "value": top["value"],
        "count": top["count"],
        "evidence": {
            "count_after_deduplication": top["count_deduped"],
            "all_candidates": candidates,
        },
    }


def detect_unit_mixup(
    train_df: pd.DataFrame,
    group_col: str,
    numeric_cols: List[str],
    sentinel_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare numeric statistics across group levels to detect unit mixups.

    If one group's median/mean differs from the other groups by a standard unit
    conversion factor (10, 12, 60, 100, 1000 or their reciprocals), flag it.
    """
    group_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    detected: Optional[Dict[str, Any]] = None

    sentinel_col = (
        sentinel_info["column"]
        if sentinel_info and sentinel_info.get("present")
        else None
    )
    sentinel_val = (
        sentinel_info["value"]
        if sentinel_info and sentinel_info.get("present")
        else None
    )

    for col in numeric_cols:
        working = train_df[[group_col, col]].dropna()
        # Exclude known sentinel values so they do not distort group averages
        if col == sentinel_col and sentinel_val is not None:
            working = working[working[col] != sentinel_val]

        grouped = working.groupby(group_col)[col].agg(["count", "min", "mean", "median", "max"])
        group_summaries[col] = {
            str(grp): {
                "count": int(row["count"]),
                "min": round(float(row["min"]), 3),
                "mean": round(float(row["mean"]), 3),
                "median": round(float(row["median"]), 3),
                "max": round(float(row["max"]), 3),
            }
            for grp, row in grouped.iterrows()
        }

        if len(grouped) < 2:
            continue

        medians = grouped["median"]
        for grp in grouped.index:
            other_median = float(medians.drop(index=grp).median())
            grp_median = float(medians.loc[grp])
            if other_median <= 0 or grp_median <= 0:
                continue

            ratio = grp_median / other_median
            # Check both multiplication factors (e.g., 1000) and division factors (e.g., 1/1000)
            for factor in STANDARD_UNIT_FACTORS:
                if abs(ratio - factor) / factor < 0.25:
                    detected = {
                        "column": col,
                        "group_value": str(grp),
                        "factor": factor,
                        "observed_median_ratio": round(ratio, 3),
                    }
                    break
                inv_factor = 1.0 / factor
                if abs(ratio - inv_factor) / inv_factor < 0.25:
                    detected = {
                        "column": col,
                        "group_value": str(grp),
                        "factor": round(inv_factor, 6),
                        "observed_median_ratio": round(ratio, 6),
                    }
                    break
            if detected is not None:
                break
        if detected is not None:
            break

    if detected is None:
        return {
            "present": False,
            "column": None,
            "group_value": None,
            "factor": None,
            "evidence": {"group_summaries": group_summaries},
        }

    return {
        "present": True,
        "column": detected["column"],
        "group_value": detected["group_value"],
        "factor": detected["factor"],
        "evidence": {
            "observed_median_ratio": detected["observed_median_ratio"],
            "group_summaries": group_summaries,
        },
    }


def detect_duplicate_rows(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Count exact extra duplicate rows in the training dataset."""
    extra_copies = int(train_df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
        "evidence": {
            "total_rows": int(len(train_df)),
            "unique_rows": int(len(train_df.drop_duplicates())),
            "extra_duplicate_rows": extra_copies,
        },
    }


def detect_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str,
    numeric_cols: List[str],
    min_run_rows: int = 20,
    min_run_days: int = 5,
) -> Dict[str, Any]:
    """Sort by date and search for long continuous runs of identical sensor values."""
    sort_cols = [date_col, "record_id"] if "record_id" in train_df.columns else [date_col]
    df_sorted = train_df.sort_values(sort_cols).reset_index(drop=True)

    longest_runs: Dict[str, Dict[str, Any]] = {}
    best_stuck: Optional[Dict[str, Any]] = None

    for col in numeric_cols:
        series = df_sorted[col]
        run_ids = (series != series.shift()).cumsum()
        runs = (
            df_sorted.groupby(run_ids)
            .agg(
                stuck_value=(col, "first"),
                row_count=(col, "size"),
                start_date=(date_col, "min"),
                end_date=(date_col, "max"),
                unique_dates=(date_col, "nunique"),
            )
            .sort_values("row_count", ascending=False)
        )
        top_run = runs.iloc[0]
        run_info = {
            "column": col,
            "stuck_value": float(top_run["stuck_value"]),
            "row_count": int(top_run["row_count"]),
            "start_date": str(top_run["start_date"]),
            "end_date": str(top_run["end_date"]),
            "unique_dates": int(top_run["unique_dates"]),
        }
        longest_runs[col] = run_info

        if (
            run_info["row_count"] >= min_run_rows
            and run_info["unique_dates"] >= min_run_days
        ):
            if best_stuck is None or run_info["row_count"] > best_stuck["row_count"]:
                best_stuck = run_info

    if best_stuck is None:
        return {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None,
            "evidence": {"longest_runs_per_column": longest_runs},
        }

    return {
        "present": True,
        "column": best_stuck["column"],
        "start_date": best_stuck["start_date"],
        "end_date": best_stuck["end_date"],
        "evidence": {
            "stuck_value": best_stuck["stuck_value"],
            "row_count": best_stuck["row_count"],
            "unique_dates": best_stuck["unique_dates"],
            "longest_runs_per_column": longest_runs,
        },
    }


def detect_target_typos(
    train_df: pd.DataFrame,
    target_col: str,
    reference_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect rows where the target value was entered 10 times too large.

    Sorts the target descending and looks for a large empty gap separating a small
    cluster of extreme values from the main distribution.
    """
    sorted_vals = train_df[target_col].dropna().sort_values(ascending=False).values
    if len(sorted_vals) < 10:
        return {"present": False, "count": None, "evidence": {}}

    # Inspect the top 5% of values for a sharp jump (ratio >= 2.0 between adjacent values)
    max_check = max(5, int(len(sorted_vals) * 0.05))
    ratios = sorted_vals[:max_check] / np.maximum(sorted_vals[1 : max_check + 1], 1e-9)
    max_gap_idx = int(np.argmax(ratios))
    max_ratio = float(ratios[max_gap_idx])

    if max_ratio < 2.0:
        return {
            "present": False,
            "count": None,
            "evidence": {
                "max_adjacent_ratio_in_top_5pct": round(max_ratio, 3),
                "top_10_values": [round(float(v), 3) for v in sorted_vals[:10]],
            },
        }

    upper_normal = float(sorted_vals[max_gap_idx + 1])
    lower_typo = float(sorted_vals[max_gap_idx])
    threshold = (upper_normal + lower_typo) / 2.0
    typo_rows = train_df[train_df[target_col] > threshold]
    count = int(len(typo_rows))

    # Cross-check ratio against reference column if available (e.g. monthly bill)
    affected_records = []
    for _, row in typo_rows.iterrows():
        entry = {
            "record_id": str(row["record_id"]) if "record_id" in row else None,
            "date": str(row["date"]) if "date" in row else None,
            target_col: round(float(row[target_col]), 3),
        }
        if reference_col and reference_col in row and row[reference_col] > 0:
            entry[reference_col] = round(float(row[reference_col]), 3)
            entry["ratio_to_reference"] = round(
                float(row[target_col] / row[reference_col]), 3
            )
        affected_records.append(entry)

    return {
        "present": count > 0,
        "count": count if count > 0 else None,
        "evidence": {
            "max_normal_value": round(upper_normal, 3),
            "min_typo_value": round(lower_typo, 3),
            "gap_ratio": round(max_ratio, 3),
            "count_after_deduplication": int(
                (train_df.drop_duplicates()[target_col] > threshold).sum()
            ),
            "affected_records": affected_records,
        },
    }


def _normalize_label(value: str) -> str:
    """Normalize text category by lowercasing and removing spaces/punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def detect_spelling_variants(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Check categorical columns for inconsistent spellings compared to later.csv."""
    inspection: Dict[str, Any] = {}
    flagged: Optional[Dict[str, Any]] = None

    for col in categorical_cols:
        train_levels = sorted(train_df[col].dropna().astype(str).unique().tolist())
        later_levels = sorted(later_df[col].dropna().astype(str).unique().tolist())
        extra_in_train = sorted(set(train_levels) - set(later_levels))
        normalized_train = {_normalize_label(v) for v in train_levels}

        inspection[col] = {
            "train_unique": train_levels,
            "later_unique": later_levels,
            "extra_spellings_in_train": extra_in_train,
            "normalized_unique_count": len(normalized_train),
        }

        if extra_in_train or len(train_levels) != len(normalized_train):
            flagged = {
                "column": col,
                "true_levels": len(later_levels),
            }

    if flagged is None:
        return {
            "present": False,
            "column": None,
            "true_levels": None,
            "evidence": {"columns_checked": inspection},
        }

    return {
        "present": True,
        "column": flagged["column"],
        "true_levels": flagged["true_levels"],
        "evidence": {"columns_checked": inspection},
    }


def run_all_quality_checks(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "gb_per_day",
    group_col: str = "circle",
    date_col: str = "date",
) -> Dict[str, Dict[str, Any]]:
    """Run all seven hidden data quality checks on train.csv."""
    leaking = detect_leaking_column(train_df, holdout_df, target_col=target_col)
    leaking_col = leaking["column"] if leaking["present"] else None

    feature_numeric_cols = [
        col
        for col in train_df.select_dtypes(include=[np.number]).columns
        if col not in {target_col, leaking_col}
    ]
    categorical_cols = [
        col
        for col in train_df.select_dtypes(include=["object", "category"]).columns
        if col not in {"record_id", date_col}
    ]

    sentinel = detect_sentinel_value(train_df, numeric_cols=feature_numeric_cols)
    unit_mixup = detect_unit_mixup(
        train_df,
        group_col=group_col,
        numeric_cols=feature_numeric_cols,
        sentinel_info=sentinel,
    )
    duplicates = detect_duplicate_rows(train_df)
    stuck_sensor = detect_stuck_sensor(
        train_df,
        date_col=date_col,
        numeric_cols=feature_numeric_cols,
    )
    target_typos = detect_target_typos(
        train_df,
        target_col=target_col,
        reference_col=leaking_col,
    )
    spelling_variants = detect_spelling_variants(
        train_df,
        later_df=later_df,
        categorical_cols=categorical_cols,
    )

    return {
        "leaking_column": leaking,
        "unit_mixup": unit_mixup,
        "sentinel_value": sentinel,
        "duplicate_rows": duplicates,
        "stuck_sensor": stuck_sensor,
        "target_typos": target_typos,
        "spelling_variants": spelling_variants,
    }
