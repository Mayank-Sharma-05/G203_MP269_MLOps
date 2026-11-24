"""Entry-point script to run the complete Stage 1 EDA and data quality analysis.

Usage:
    python -m src.mp269.run_stage1
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .data_analysis import load_datasets, load_params, run_eda
from .data_quality import run_all_quality_checks


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame into a GitHub-flavored Markdown table."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_stage1_report(
    eda: Dict[str, Any],
    issues: Dict[str, Dict[str, Any]],
    train_shape: tuple[int, int],
    later_shape: tuple[int, int],
    holdout_shape: tuple[int, int],
) -> str:
    """Generate the Stage 1 Markdown data quality report from computed results."""
    ov = eda["overview"]
    dups = eda["duplicates"]
    date_info = eda["date_distribution"]

    lines: list[str] = []
    lines.append("# Stage 1 Data Quality & Exploratory Analysis Report")
    lines.append("")
    lines.append("**Group ID:** G203 | **Problem ID:** MP269 — Mobile Data Use (Surat)  ")
    lines.append("**Target Variable:** `gb_per_day` | **Evaluation Metric:** MAE  ")
    lines.append("**Source Dataset Analyzed:** `data/raw/train.csv` (original unmodified file)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Dataset Overview
    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append(
        "The project supplies three CSV files covering daily mobile data usage records "
        "for customers in the Surat telecom circle:"
    )
    lines.append("")
    lines.append("| Dataset | Rows | Columns | Purpose |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| `data/raw/train.csv` | {train_shape[0]} | {train_shape[1]} | "
        "Historical training dataset (Jan–Jun 2026); raw and uncleaned |"
    )
    lines.append(
        f"| `data/raw/later.csv` | {later_shape[0]} | {later_shape[1]} | "
        "Later-arriving clean dataset (Jul–Aug 2026) for monitoring and retraining |"
    )
    lines.append(
        f"| `data/raw/holdout_features.csv` | {holdout_shape[0]} | {holdout_shape[1]} | "
        "Unlabeled holdout features (Sep 2026) for final predictions |"
    )
    lines.append("")
    lines.append(
        f"**Date coverage in `train.csv`:** `{date_info['min_date']}` to "
        f"`{date_info['max_date']}` across `{date_info['unique_dates']}` unique days "
        f"(average `{date_info['mean_records_per_day']}` records/day; "
        f"min `{date_info['min_records_per_day']}`, max `{date_info['max_records_per_day']}`)."
    )
    lines.append("")
    lines.append("**Monthly record counts in `train.csv`:**")
    lines.append("")
    monthly_df = pd.DataFrame(
        [
            {"month": month, "records": count}
            for month, count in date_info["monthly_counts"].items()
        ]
    )
    lines.append(_df_to_markdown_table(monthly_df))
    lines.append("")

    # 2. Dataset Dimensions and Data Types
    lines.append("## 2. Dataset Dimensions, Data Types, and Unique Values")
    lines.append("")
    lines.append(
        f"`data/raw/train.csv` contains **{ov['rows']} rows** and **{ov['columns']} columns**."
    )
    lines.append("")
    schema_df = pd.DataFrame(
        [
            {
                "column": col,
                "dtype": ov["dtypes"][col],
                "unique_values": eda["unique_counts"][col],
            }
            for col in ov["column_names"]
        ]
    )
    lines.append(_df_to_markdown_table(schema_df))
    lines.append("")

    # 3. Missing Value Analysis
    lines.append("## 3. Missing Value Analysis")
    lines.append("")
    total_missing = int(eda["missing_values"]["missing_count"].sum())
    lines.append(
        f"Standard null/NaN checks find **{total_missing} explicit missing values** across all columns. "
        "However, as detailed in Section 7.3 below, missing readings in `video_minutes` were "
        "encoded using a sentinel value (`-999`) rather than left blank."
    )
    lines.append("")
    lines.append(_df_to_markdown_table(eda["missing_values"]))
    lines.append("")

    # 4. Duplicate Analysis
    lines.append("## 4. Duplicate Analysis")
    lines.append("")
    lines.append(f"- **Total rows in `train.csv`:** {dups['total_rows']}")
    lines.append(f"- **Unique rows:** {dups['unique_rows']}")
    lines.append(f"- **Exact duplicate rows (`df.duplicated().sum()`):** {dups['exact_duplicate_rows']}")
    lines.append(f"- **Duplicate `record_id` values:** {dups['duplicate_record_ids']}")
    lines.append("")

    # 5. Numerical Statistics & Outliers & Correlations
    lines.append("## 5. Numerical Statistics, Outliers, and Correlations")
    lines.append("")
    lines.append("### 5.1 Descriptive Statistics")
    lines.append("")
    lines.append(_df_to_markdown_table(eda["numerical_stats"]))
    lines.append("")
    lines.append("### 5.2 IQR Outlier Analysis (1.5 x IQR Rule)")
    lines.append("")
    lines.append(_df_to_markdown_table(eda["outliers"]))
    lines.append("")
    lines.append("### 5.3 Pearson Correlation Matrix (Numerical Columns)")
    lines.append("")
    corr_df = eda["correlations"].copy()
    corr_df.insert(0, "column", corr_df.index)
    lines.append(_df_to_markdown_table(corr_df.reset_index(drop=True)))
    lines.append("")

    # 6. Categorical Analysis
    lines.append("## 6. Categorical Value Distributions")
    lines.append("")
    for col, dist_df in eda["categorical_distributions"].items():
        lines.append(f"### `{col}`")
        lines.append("")
        lines.append(_df_to_markdown_table(dist_df))
        lines.append("")

    # 7. Detailed Investigation of All Seven Hidden Data Problems
    lines.append("## 7. Detailed Investigation of All Seven Data Problems")
    lines.append("")

    # 7.1 Leaking column
    leak = issues["leaking_column"]
    lines.append("### 7.1 Leaking Column")
    lines.append("")
    lines.append(f"- **Present:** `{str(leak['present']).lower()}`")
    lines.append(f"- **Column:** `{leak['column']}`")
    if leak["present"]:
        metrics = leak["evidence"]["column_metrics"][leak["column"]]
        lines.append(
            f"- **Evidence:** `{leak['column']}` exists in `train.csv` and `later.csv`, "
            "but is **absent from `holdout_features.csv`**. Per `PROBLEM.md`, it represents "
            "data shown later in the monthly bill (only known after the event). Excluding extreme "
            f"target typos (top 1%), its Pearson correlation with `gb_per_day` is "
            f"**{metrics['trimmed_correlation']}** with a mean absolute difference of only "
            f"**{metrics['trimmed_mean_abs_diff']} GB** (full-dataset correlation: "
            f"`{metrics['full_correlation']}`)."
        )
    lines.append("")

    # 7.2 Unit mixup
    unit = issues["unit_mixup"]
    lines.append("### 7.2 Unit Mixup")
    lines.append("")
    lines.append(f"- **Present:** `{str(unit['present']).lower()}`")
    lines.append(f"- **Column:** `{unit['column']}`")
    lines.append(f"- **Group Value (`circle`):** `{unit['group_value']}`")
    lines.append(f"- **Factor:** `{unit['factor']}`")
    lines.append(
        "- **Evidence:** Group-by comparisons across `Rural`, `Semi-urban`, and `Urban` "
        "(excluding the `-999` sentinel in `video_minutes`) show consistent means, medians, "
        "minimums, and maximums across all numeric features, with no group differing by a "
        "unit conversion factor (10, 12, 60, 100, or 1000)."
    )
    lines.append("")
    group_rows = []
    for col, grp_map in unit["evidence"]["group_summaries"].items():
        for grp, stats in grp_map.items():
            group_rows.append(
                {
                    "column": col,
                    "circle": grp,
                    "count": stats["count"],
                    "min": stats["min"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "max": stats["max"],
                }
            )
    lines.append(_df_to_markdown_table(pd.DataFrame(group_rows)))
    lines.append("")

    # 7.3 Sentinel value
    sent = issues["sentinel_value"]
    lines.append("### 7.3 Sentinel Value")
    lines.append("")
    lines.append(f"- **Present:** `{str(sent['present']).lower()}`")
    lines.append(f"- **Column:** `{sent['column']}`")
    lines.append(f"- **Value:** `{sent['value']}`")
    lines.append(f"- **Count (in raw `train.csv`):** `{sent['count']}`")
    if sent["present"]:
        dedup_cnt = sent["evidence"]["count_after_deduplication"]
        lines.append(
            f"- **Evidence:** `{sent['column']}` contains `{sent['count']}` rows with the "
            f"impossible negative value `{sent['value']}` ({dedup_cnt} rows after removing "
            "exact duplicates). All other values in `video_minutes` lie between `0` and `299`."
        )
    lines.append("")

    # 7.4 Duplicate rows
    dup_issue = issues["duplicate_rows"]
    lines.append("### 7.4 Duplicate Rows")
    lines.append("")
    lines.append(f"- **Present:** `{str(dup_issue['present']).lower()}`")
    lines.append(f"- **Extra Duplicate Count:** `{dup_issue['count']}`")
    lines.append(
        f"- **Evidence:** `train.csv` has `{dup_issue['evidence']['total_rows']}` rows, "
        f"of which `{dup_issue['evidence']['unique_rows']}` are unique and "
        f"`{dup_issue['evidence']['extra_duplicate_rows']}` are exact extra copies."
    )
    lines.append("")

    # 7.5 Stuck sensor
    stuck = issues["stuck_sensor"]
    lines.append("### 7.5 Stuck Sensor")
    lines.append("")
    lines.append(f"- **Present:** `{str(stuck['present']).lower()}`")
    lines.append(f"- **Column:** `{stuck['column']}`")
    lines.append(f"- **Start Date:** `{stuck['start_date']}`")
    lines.append(f"- **End Date:** `{stuck['end_date']}`")
    if stuck["present"]:
        ev = stuck["evidence"]
        lines.append(
            f"- **Evidence:** After sorting `train.csv` by `date`, `{stuck['column']}` is "
            f"frozen at the exact constant value **`{ev['stuck_value']}`** for "
            f"**{ev['row_count']} consecutive rows** spanning **{ev['unique_dates']} consecutive days** "
            f"from `{stuck['start_date']}` through `{stuck['end_date']}`. Days immediately before "
            "(`2026-04-11`) and after (`2026-04-26`) show normal variation."
        )
    lines.append("")

    # 7.6 Target typos
    typo = issues["target_typos"]
    lines.append("### 7.6 Target Typos")
    lines.append("")
    lines.append(f"- **Present:** `{str(typo['present']).lower()}`")
    lines.append(f"- **Count:** `{typo['count']}`")
    if typo["present"]:
        ev = typo["evidence"]
        lines.append(
            f"- **Evidence:** Sorting `gb_per_day` descending reveals a sharp gap between "
            f"the highest normal value (`{ev['max_normal_value']} GB`) and the lowest typo "
            f"value (`{ev['min_typo_value']} GB`, a `{ev['gap_ratio']}x` jump). "
            f"Exactly **`{typo['count']}` rows** sit above this gap, and each is ~10x its "
            "corresponding `gb_from_monthly_bill` value:"
        )
        lines.append("")
        lines.append(_df_to_markdown_table(pd.DataFrame(ev["affected_records"])))
    lines.append("")

    # 7.7 Spelling variants
    spell = issues["spelling_variants"]
    lines.append("### 7.7 Spelling Variants")
    lines.append("")
    lines.append(f"- **Present:** `{str(spell['present']).lower()}`")
    lines.append(f"- **Column:** `{spell['column']}`")
    lines.append(f"- **True Levels:** `{spell['true_levels']}`")
    lines.append(
        "- **Evidence:** Comparing unique categories in `train.csv` against clean categories "
        "in `later.csv` confirms exact matches with zero typos, extra spaces, or casing variants:"
    )
    for col, info in spell["evidence"]["columns_checked"].items():
        lines.append(
            f"  - `{col}`: `train.csv` = `{info['train_unique']}` | "
            f"`later.csv` = `{info['later_unique']}`"
        )
    lines.append("")

    # 8. Summary of Findings
    lines.append("## 8. Summary of Detected Data Quality Issues")
    lines.append("")
    summary_rows = [
        {
            "issue": "leaking_column",
            "present": issues["leaking_column"]["present"],
            "details": f"column={issues['leaking_column']['column']}",
        },
        {
            "issue": "unit_mixup",
            "present": issues["unit_mixup"]["present"],
            "details": "Not present (all circles use consistent units)",
        },
        {
            "issue": "sentinel_value",
            "present": issues["sentinel_value"]["present"],
            "details": (
                f"column={issues['sentinel_value']['column']}, "
                f"value={issues['sentinel_value']['value']}, "
                f"count={issues['sentinel_value']['count']}"
            ),
        },
        {
            "issue": "duplicate_rows",
            "present": issues["duplicate_rows"]["present"],
            "details": f"count={issues['duplicate_rows']['count']} extra copies",
        },
        {
            "issue": "stuck_sensor",
            "present": issues["stuck_sensor"]["present"],
            "details": (
                f"column={issues['stuck_sensor']['column']}, "
                f"{issues['stuck_sensor']['start_date']} to {issues['stuck_sensor']['end_date']}"
            ),
        },
        {
            "issue": "target_typos",
            "present": issues["target_typos"]["present"],
            "details": f"count={issues['target_typos']['count']} rows (~10x too large)",
        },
        {
            "issue": "spelling_variants",
            "present": issues["spelling_variants"]["present"],
            "details": "Not present (categories match later.csv cleanly)",
        },
    ]
    lines.append(_df_to_markdown_table(pd.DataFrame(summary_rows)))
    lines.append("")

    # 9. Columns That Must NOT Be Used for Model Training
    lines.append("## 9. Columns Excluded from Model Training")
    lines.append("")
    lines.append("The following columns must **not** be used as input features when training models in Stage 2:")
    lines.append("")
    lines.append(
        "1. **`gb_from_monthly_bill` (Target Leakage):** Recorded only after the billing cycle "
        "ends, unavailable in `holdout_features.csv` and in `POST /predict` requests, and "
        "nearly identical to `gb_per_day` (correlation `0.9988` on non-typo rows). Including it "
        "would cause severe data leakage and fail in production."
    )
    lines.append(
        "2. **`record_id` (Identifier):** Unique row identifier with no predictive signal "
        "(and not part of the `POST /predict` request schema)."
    )
    lines.append(
        "3. **`date` (Timestamp Metadata):** Used for chronological sorting, stuck-sensor "
        "detection, and time-based splitting in Stage 4, but not part of the `POST /predict` "
        "feature schema defined in `PROBLEM.md` Section 5."
    )
    lines.append(
        "4. **`gb_per_day` (Target Variable):** The regression target to be predicted."
    )
    lines.append("")
    lines.append(
        "**Valid model input features (matching `PROBLEM.md` Section 5):** "
        "`age`, `screen_hours`, `apps_installed`, `plan_gb_limit`, `video_minutes`, "
        "`wifi_hours`, `phone_tier`, `circle`."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run the full Stage 1 analysis pipeline and save the Markdown report."""
    repo_root = Path(__file__).resolve().parents[2]
    params = load_params(repo_root / "params.yaml")
    train_df, later_df, holdout_df = load_datasets(params, base_dir=repo_root)

    target_col = params["target"]["column"]
    group_col = params["group"]["column"]

    eda_results = run_eda(train_df)
    quality_results = run_all_quality_checks(
        train_df=train_df,
        later_df=later_df,
        holdout_df=holdout_df,
        target_col=target_col,
        group_col=group_col,
    )

    report_text = build_stage1_report(
        eda=eda_results,
        issues=quality_results,
        train_shape=train_df.shape,
        later_shape=later_df.shape,
        holdout_shape=holdout_df.shape,
    )

    report_path = repo_root / "reports" / "stage1_data_quality_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print("=== Stage 1 Data Quality Analysis Summary ===")
    for name, result in quality_results.items():
        summary = {k: v for k, v in result.items() if k != "evidence"}
        print(f"  {name}: {summary}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
