# Stage 1 Data Quality & Exploratory Analysis Report

**Group ID:** G203 | **Problem ID:** MP269 — Mobile Data Use (Surat)  
**Target Variable:** `gb_per_day` | **Evaluation Metric:** MAE  
**Source Dataset Analyzed:** `data/raw/train.csv` (original unmodified file)

---

## 1. Dataset Overview

The project supplies three CSV files covering daily mobile data usage records for customers in the Surat telecom circle:

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| `data/raw/train.csv` | 1053 | 12 | Historical training dataset (Jan–Jun 2026); raw and uncleaned |
| `data/raw/later.csv` | 300 | 12 | Later-arriving clean dataset (Jul–Aug 2026) for monitoring and retraining |
| `data/raw/holdout_features.csv` | 250 | 10 | Unlabeled holdout features (Sep 2026) for final predictions |

**Date coverage in `train.csv`:** `2026-01-01` to `2026-06-30` across `181` unique days (average `5.82` records/day; min `1`, max `15`).

**Monthly record counts in `train.csv`:**

| month | records |
| --- | --- |
| 2026-01 | 209 |
| 2026-02 | 139 |
| 2026-03 | 153 |
| 2026-04 | 182 |
| 2026-05 | 198 |
| 2026-06 | 172 |

## 2. Dataset Dimensions, Data Types, and Unique Values

`data/raw/train.csv` contains **1053 rows** and **12 columns**.

| column | dtype | unique_values |
| --- | --- | --- |
| record_id | object | 1000 |
| date | object | 181 |
| age | int64 | 55 |
| screen_hours | float64 | 116 |
| apps_installed | int64 | 190 |
| plan_gb_limit | float64 | 41 |
| video_minutes | int64 | 286 |
| wifi_hours | float64 | 161 |
| phone_tier | object | 3 |
| circle | object | 3 |
| gb_from_monthly_bill | float64 | 887 |
| gb_per_day | float64 | 905 |

## 3. Missing Value Analysis

Standard null/NaN checks find **0 explicit missing values** across all columns. However, as detailed in Section 7.3 below, missing readings in `video_minutes` were encoded using a sentinel value (`-999`) rather than left blank.

| column | missing_count | missing_pct |
| --- | --- | --- |
| record_id | 0 | 0 |
| date | 0 | 0 |
| age | 0 | 0 |
| screen_hours | 0 | 0 |
| apps_installed | 0 | 0 |
| plan_gb_limit | 0 | 0 |
| video_minutes | 0 | 0 |
| wifi_hours | 0 | 0 |
| phone_tier | 0 | 0 |
| circle | 0 | 0 |
| gb_from_monthly_bill | 0 | 0 |
| gb_per_day | 0 | 0 |

## 4. Duplicate Analysis

- **Total rows in `train.csv`:** 1053
- **Unique rows:** 1000
- **Exact duplicate rows (`df.duplicated().sum()`):** 53
- **Duplicate `record_id` values:** 53

## 5. Numerical Statistics, Outliers, and Correlations

### 5.1 Descriptive Statistics

| column | count | mean | std | min | 25% | median | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1053 | 42.2213 | 15.5687 | 16 | 29 | 42 | 55 | 70 |
| screen_hours | 1053 | 6.2552 | 3.3456 | 0.5 | 3.4 | 6.3 | 9.2 | 12 |
| apps_installed | 1053 | 106.283 | 54.1741 | 10 | 61 | 107 | 152 | 200 |
| plan_gb_limit | 1053 | 2.9808 | 1.155 | 1 | 2 | 3 | 3.9 | 5 |
| video_minutes | 1053 | 93.1425 | 260.8623 | -999 | 60 | 145 | 222 | 299 |
| wifi_hours | 1053 | 7.8683 | 4.4144 | 0 | 4.7 | 7.7 | 11.6 | 16 |
| gb_from_monthly_bill | 1053 | 3.882 | 1.2764 | 0.119 | 2.98 | 3.865 | 4.783 | 7.823 |
| gb_per_day | 1053 | 4.1366 | 3.2855 | 0.057 | 2.984 | 3.883 | 4.822 | 50.75 |

### 5.2 IQR Outlier Analysis (1.5 x IQR Rule)

| column | q1 | q3 | iqr | lower_bound | upper_bound | low_outliers | high_outliers | total_outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 29 | 55 | 26 | -10 | 94 | 0 | 0 | 0 |
| screen_hours | 3.4 | 9.2 | 5.8 | -5.3 | 17.9 | 0 | 0 | 0 |
| apps_installed | 61 | 152 | 91 | -75.5 | 288.5 | 0 | 0 | 0 |
| plan_gb_limit | 2 | 3.9 | 1.9 | -0.85 | 6.75 | 0 | 0 | 0 |
| video_minutes | 60 | 222 | 162 | -183 | 465 | 51 | 0 | 51 |
| wifi_hours | 4.7 | 11.6 | 6.9 | -5.65 | 21.95 | 0 | 0 | 0 |
| gb_from_monthly_bill | 2.98 | 4.783 | 1.803 | 0.275 | 7.488 | 1 | 2 | 3 |
| gb_per_day | 2.984 | 4.822 | 1.838 | 0.227 | 7.579 | 1 | 10 | 11 |

### 5.3 Pearson Correlation Matrix (Numerical Columns)

| column | age | screen_hours | apps_installed | plan_gb_limit | video_minutes | wifi_hours | gb_from_monthly_bill | gb_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1 | 0.0569 | 0.0359 | 0.0077 | -0.0676 | -0.0276 | -0.272 | -0.1367 |
| screen_hours | 0.0569 | 1 | 0.0295 | -0.0318 | -0.0304 | -0.0122 | 0.7494 | 0.2812 |
| apps_installed | 0.0359 | 0.0295 | 1 | -0.021 | -0.0518 | 0.0177 | 0.0491 | 0.0269 |
| plan_gb_limit | 0.0077 | -0.0318 | -0.021 | 1 | -0.0488 | -0.0225 | 0.1444 | 0.0425 |
| video_minutes | -0.0676 | -0.0304 | -0.0518 | -0.0488 | 1 | 0.0276 | 0.09 | 0.006 |
| wifi_hours | -0.0276 | -0.0122 | 0.0177 | -0.0225 | 0.0276 | 1 | -0.2865 | -0.0769 |
| gb_from_monthly_bill | -0.272 | 0.7494 | 0.0491 | 0.1444 | 0.09 | -0.2865 | 1 | 0.397 |
| gb_per_day | -0.1367 | 0.2812 | 0.0269 | 0.0425 | 0.006 | -0.0769 | 0.397 | 1 |

## 6. Categorical Value Distributions

### `phone_tier`

| category | count | percentage |
| --- | --- | --- |
| basic | 411 | 39.03 |
| flagship | 337 | 32 |
| mid | 305 | 28.96 |

### `circle`

| category | count | percentage |
| --- | --- | --- |
| Urban | 416 | 39.51 |
| Semi-urban | 365 | 34.66 |
| Rural | 272 | 25.83 |

## 7. Detailed Investigation of All Seven Data Problems

### 7.1 Leaking Column

- **Present:** `true`
- **Column:** `gb_from_monthly_bill`
- **Evidence:** `gb_from_monthly_bill` exists in `train.csv` and `later.csv`, but is **absent from `holdout_features.csv`**. Per `PROBLEM.md`, it represents data shown later in the monthly bill (only known after the event). Excluding extreme target typos (top 1%), its Pearson correlation with `gb_per_day` is **0.998814** with a mean absolute difference of only **0.049179 GB** (full-dataset correlation: `0.397011`).

### 7.2 Unit Mixup

- **Present:** `false`
- **Column:** `None`
- **Group Value (`circle`):** `None`
- **Factor:** `None`
- **Evidence:** Group-by comparisons across `Rural`, `Semi-urban`, and `Urban` (excluding the `-999` sentinel in `video_minutes`) show consistent means, medians, minimums, and maximums across all numeric features, with no group differing by a unit conversion factor (10, 12, 60, 100, or 1000).

| column | circle | count | min | mean | median | max |
| --- | --- | --- | --- | --- | --- | --- |
| age | Rural | 272 | 16 | 41.853 | 42 | 70 |
| age | Semi-urban | 365 | 16 | 41.907 | 41 | 70 |
| age | Urban | 416 | 16 | 42.738 | 43 | 70 |
| screen_hours | Rural | 272 | 0.5 | 6.137 | 6.1 | 11.9 |
| screen_hours | Semi-urban | 365 | 0.5 | 6.179 | 6.2 | 12 |
| screen_hours | Urban | 416 | 0.6 | 6.399 | 6.7 | 12 |
| apps_installed | Rural | 272 | 10 | 105.779 | 103 | 200 |
| apps_installed | Semi-urban | 365 | 10 | 105.258 | 106 | 200 |
| apps_installed | Urban | 416 | 10 | 107.512 | 107 | 200 |
| plan_gb_limit | Rural | 272 | 1 | 2.974 | 3 | 5 |
| plan_gb_limit | Semi-urban | 365 | 1 | 3.039 | 3.2 | 5 |
| plan_gb_limit | Urban | 416 | 1 | 2.934 | 2.85 | 5 |
| video_minutes | Rural | 252 | 2 | 144.595 | 143 | 299 |
| video_minutes | Semi-urban | 353 | 0 | 152.504 | 154 | 299 |
| video_minutes | Urban | 397 | 0 | 148 | 152 | 299 |
| wifi_hours | Rural | 272 | 0 | 8.054 | 8.1 | 16 |
| wifi_hours | Semi-urban | 365 | 0.1 | 7.835 | 7.6 | 16 |
| wifi_hours | Urban | 416 | 0 | 7.776 | 7.5 | 16 |

### 7.3 Sentinel Value

- **Present:** `true`
- **Column:** `video_minutes`
- **Value:** `-999`
- **Count (in raw `train.csv`):** `51`
- **Evidence:** `video_minutes` contains `51` rows with the impossible negative value `-999` (49 rows after removing exact duplicates). All other values in `video_minutes` lie between `0` and `299`.

### 7.4 Duplicate Rows

- **Present:** `true`
- **Extra Duplicate Count:** `53`
- **Evidence:** `train.csv` has `1053` rows, of which `1000` are unique and `53` are exact extra copies.

### 7.5 Stuck Sensor

- **Present:** `true`
- **Column:** `wifi_hours`
- **Start Date:** `2026-04-12`
- **End Date:** `2026-04-25`
- **Evidence:** After sorting `train.csv` by `date`, `wifi_hours` is frozen at the exact constant value **`4.7`** for **90 consecutive rows** spanning **14 consecutive days** from `2026-04-12` through `2026-04-25`. Days immediately before (`2026-04-11`) and after (`2026-04-26`) show normal variation.

### 7.6 Target Typos

- **Present:** `true`
- **Count:** `8`
- **Evidence:** Sorting `gb_per_day` descending reveals a sharp gap between the highest normal value (`7.84 GB`) and the lowest typo value (`23.39 GB`, a `2.983x` jump). Exactly **`8` rows** sit above this gap, and each is ~10x its corresponding `gb_from_monthly_bill` value:

| record_id | date | gb_per_day | gb_from_monthly_bill | ratio_to_reference |
| --- | --- | --- | --- | --- |
| R00055 | 2026-01-08 | 35.46 | 3.518 | 10.08 |
| R00313 | 2026-02-24 | 29.27 | 2.927 | 10 |
| R00411 | 2026-03-17 | 24.74 | 2.48 | 9.976 |
| R00558 | 2026-04-15 | 38.57 | 3.863 | 9.984 |
| R00644 | 2026-04-30 | 48.62 | 4.842 | 10.041 |
| R00838 | 2026-06-02 | 50.75 | 5.144 | 9.866 |
| R00932 | 2026-06-18 | 46.93 | 4.749 | 9.882 |
| R00951 | 2026-06-22 | 23.39 | 2.386 | 9.803 |

### 7.7 Spelling Variants

- **Present:** `false`
- **Column:** `None`
- **True Levels:** `None`
- **Evidence:** Comparing unique categories in `train.csv` against clean categories in `later.csv` confirms exact matches with zero typos, extra spaces, or casing variants:
  - `phone_tier`: `train.csv` = `['basic', 'flagship', 'mid']` | `later.csv` = `['basic', 'flagship', 'mid']`
  - `circle`: `train.csv` = `['Rural', 'Semi-urban', 'Urban']` | `later.csv` = `['Rural', 'Semi-urban', 'Urban']`

## 8. Summary of Detected Data Quality Issues

| issue | present | details |
| --- | --- | --- |
| leaking_column | True | column=gb_from_monthly_bill |
| unit_mixup | False | Not present (all circles use consistent units) |
| sentinel_value | True | column=video_minutes, value=-999, count=51 |
| duplicate_rows | True | count=53 extra copies |
| stuck_sensor | True | column=wifi_hours, 2026-04-12 to 2026-04-25 |
| target_typos | True | count=8 rows (~10x too large) |
| spelling_variants | False | Not present (categories match later.csv cleanly) |

## 9. Columns Excluded from Model Training

The following columns must **not** be used as input features when training models in Stage 2:

1. **`gb_from_monthly_bill` (Target Leakage):** Recorded only after the billing cycle ends, unavailable in `holdout_features.csv` and in `POST /predict` requests, and nearly identical to `gb_per_day` (correlation `0.9988` on non-typo rows). Including it would cause severe data leakage and fail in production.
2. **`record_id` (Identifier):** Unique row identifier with no predictive signal (and not part of the `POST /predict` request schema).
3. **`date` (Timestamp Metadata):** Used for chronological sorting, stuck-sensor detection, and time-based splitting in Stage 4, but not part of the `POST /predict` feature schema defined in `PROBLEM.md` Section 5.
4. **`gb_per_day` (Target Variable):** The regression target to be predicted.

**Valid model input features (matching `PROBLEM.md` Section 5):** `age`, `screen_hours`, `apps_installed`, `plan_gb_limit`, `video_minutes`, `wifi_hours`, `phone_tier`, `circle`.
