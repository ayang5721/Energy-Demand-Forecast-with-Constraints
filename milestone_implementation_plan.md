# Implementation Plan: CS229 Milestone Only

## Current Repo Assumption

Your repo currently has this structure:

```text
repo/
└── data/
    └── hrl_load_metered.csv
```

The file `data/hrl_load_metered.csv` is the only input data file.

The uploaded file has this format:

```text
datetime_beginning_utc, datetime_beginning_ept, nerc_region, mkt_region, zone, load_area, mw, is_verified
```

Dataset facts from the provided CSV:

```text
Rows: 2880
Zone: AEP
Market region: WEST
NERC region: RFC
Load areas: AEPAPT, AEPIMP, AEPKPT, AEPOPT
Rows per load area: 720
Unique hourly timestamps: 720
UTC range: 2026-04-01 04:00:00 to 2026-05-01 03:00:00
All rows verified: True
```

Each raw row is:

```text
one load area × one hour → MW load
```

The milestone task is:

```text
Use features at time t to predict load_mw at time t + 24 hours
```

This is a 24-hour-ahead load forecasting problem.

---

# 1. Milestone Deliverable to Implement

Build an end-to-end Python pipeline that:

```text
1. Loads data/hrl_load_metered.csv.
2. Cleans and validates the data.
3. Builds supervised ML features.
4. Splits data chronologically.
5. Trains three models:
      - Persistence baseline
      - Ordinary Least Squares
      - Ridge regression
6. Evaluates models using:
      - MAE
      - RMSE
      - MAPE
      - Bias
7. Saves prediction and metric CSVs.
8. Creates two milestone plots:
      - true vs predicted sample plot
      - error by hour plot
9. Aggregates load-area predictions into total AEP zone load.
10. Computes under-generation and over-generation metrics.
```

Do **not** implement final-project-only work yet.

Do **not** implement:

```text
lasso
neural network
PCA
dispatch optimizer
reserve-margin experiment
LMP/cost data
report generation
```

---

# 2. Required Repo Structure to Create

Create this structure:

```text
repo/
├── data/
│   └── hrl_load_metered.csv
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── split.py
│   ├── models.py
│   ├── evaluate.py
│   ├── plots.py
│   ├── operational.py
│   └── run_milestone.py
│
├── results/
│   └── milestone/
│       ├── metrics/
│       ├── predictions/
│       └── figures/
│
├── requirements.txt
└── README.md
```

Use the existing `data/hrl_load_metered.csv` path exactly.

---

# 3. Install Dependencies

Use the accompanying `requirements.txt`.

Install with:

```bash
pip install -r requirements.txt
```

Dependencies should be:

```text
pandas
numpy
scikit-learn
matplotlib
pyyaml
```

No paid API or external data is needed.

---

# 4. Implement `src/data.py`

## Purpose

Load and clean the raw PJM CSV.

## Required functions

### `load_raw_data(path: str) -> pd.DataFrame`

Behavior:

```text
read CSV from path
return raw dataframe
```

### `clean_pjm_data(df: pd.DataFrame) -> pd.DataFrame`

Input raw columns:

```text
datetime_beginning_utc
datetime_beginning_ept
nerc_region
mkt_region
zone
load_area
mw
is_verified
```

Output columns:

```text
timestamp_utc
timestamp_ept
nerc_region
market_region
zone
load_area
load_mw
is_verified
```

Implementation details:

```text
1. Copy dataframe.
2. Standardize column names by stripping whitespace and lowercasing.
3. Parse datetime_beginning_utc into timestamp_utc.
4. Parse datetime_beginning_ept into timestamp_ept.
5. Rename mkt_region to market_region.
6. Rename mw to load_mw.
7. Convert load_mw to numeric.
8. Convert is_verified to boolean if needed.
9. Strip whitespace from string columns.
10. Keep only zone == "AEP".
11. Drop rows with missing timestamp_utc, timestamp_ept, load_area, zone, or load_mw.
12. Drop duplicates by timestamp_utc + zone + load_area.
13. Sort by timestamp_utc and load_area.
14. Reset index.
```

Expected result:

```text
2880 rows
4 load areas
720 rows per load area
```

### `validate_clean_data(df: pd.DataFrame) -> dict`

Return a dictionary with:

```text
n_rows
n_unique_timestamps
zone_values
load_area_values
rows_per_load_area
min_timestamp_utc
max_timestamp_utc
all_verified
duplicate_count_after_cleaning
missing_values_by_column
```

Also print a readable summary.

Validation expectations:

```text
n_rows should be 2880
n_unique_timestamps should be 720
zone_values should be ["AEP"]
load_area_values should be ["AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT"]
each load area should have 720 rows
all_verified should be True
```

Do not crash if values differ; print warnings instead.

---

# 5. Implement `src/features.py`

## Purpose

Create the supervised learning dataset.

The target is:

```text
target_load_mw = load_mw for the same load_area 24 hours later
```

## Required functions

### `add_time_features(df: pd.DataFrame) -> pd.DataFrame`

Use `timestamp_ept` for time features.

Add:

```text
hour
day_of_week
month
is_weekend
```

Definitions:

```text
hour = timestamp_ept hour
day_of_week = timestamp_ept dayofweek, Monday=0, Sunday=6
month = timestamp_ept month
is_weekend = 1 if day_of_week is 5 or 6 else 0
```

### `add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame`

Add:

```text
sin_hour
cos_hour
sin_day_of_week
cos_day_of_week
```

Formulas:

```text
sin_hour = sin(2π * hour / 24)
cos_hour = cos(2π * hour / 24)
sin_day_of_week = sin(2π * day_of_week / 7)
cos_day_of_week = cos(2π * day_of_week / 7)
```

### `add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame`

For milestone use:

```text
lags = [1, 24, 48]
```

For each load area separately, create:

```text
load_lag_1
load_lag_24
load_lag_48
```

Implementation detail:

```text
sort by load_area and timestamp_utc
group by load_area
shift load_mw by each lag
```

Do not compute lags across load areas.

### `add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame`

For milestone use:

```text
windows = [24]
```

For each load area separately, create:

```text
rolling_mean_24
rolling_std_24
```

Implementation detail:

```text
sort by load_area and timestamp_utc
group by load_area
rolling mean/std on load_mw
window = 24
min_periods = 24
```

Use current and prior values only. Do not use future values.

### `add_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame`

For each load area separately, create:

```text
target_load_mw
target_timestamp_utc
target_timestamp_ept
```

Implementation:

```text
target_load_mw = groupby(load_area)["load_mw"].shift(-24)
target_timestamp_utc = groupby(load_area)["timestamp_utc"].shift(-24)
target_timestamp_ept = groupby(load_area)["timestamp_ept"].shift(-24)
```

### `make_feature_dataset(df: pd.DataFrame) -> pd.DataFrame`

Run the full milestone feature pipeline:

```text
1. add_time_features
2. add_cyclical_features
3. add_lag_features with [1, 24, 48]
4. add_rolling_features with [24]
5. add_target with horizon=24
6. drop rows missing any required features or target
7. sort by timestamp_utc and load_area
8. reset index
```

Required feature columns:

```text
load_area
hour
day_of_week
month
is_weekend
sin_hour
cos_hour
sin_day_of_week
cos_day_of_week
load_mw
load_lag_1
load_lag_24
load_lag_48
rolling_mean_24
rolling_std_24
```

Required target/metadata columns:

```text
timestamp_utc
timestamp_ept
target_timestamp_utc
target_timestamp_ept
zone
load_area
target_load_mw
```

Expected row count:

```text
2592 rows
```

Reason:

```text
(720 hours - 48 max lag - 24 target horizon) × 4 load areas = 2592
```

---

# 6. Implement `src/split.py`

## Purpose

Split data chronologically by timestamp.

Do not randomly split.

## Required functions

### `time_based_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15)`

Implementation:

```text
1. Get sorted unique timestamp_utc values from feature dataframe.
2. train timestamps = first 70%.
3. validation timestamps = next 15%.
4. test timestamps = final 15%.
5. Return train_df, val_df, test_df.
```

Important:

```text
All four load areas from the same timestamp must be in the same split.
```

### `get_feature_target_metadata(df: pd.DataFrame)`

Return:

```text
X
y
metadata
```

Feature columns:

```text
load_area
hour
day_of_week
month
is_weekend
sin_hour
cos_hour
sin_day_of_week
cos_day_of_week
load_mw
load_lag_1
load_lag_24
load_lag_48
rolling_mean_24
rolling_std_24
```

Target:

```text
target_load_mw
```

Metadata:

```text
timestamp_utc
timestamp_ept
target_timestamp_utc
target_timestamp_ept
zone
load_area
load_mw
target_load_mw
```

### `validate_split(train_df, val_df, test_df)`

Print and/or assert:

```text
max train timestamp < min validation timestamp
max validation timestamp < min test timestamp
no timestamp appears in more than one split
```

---

# 7. Implement `src/models.py`

## Purpose

Train milestone models.

Models:

```text
Persistence baseline
OLS
Ridge
```

Use scikit-learn pipelines for OLS and Ridge.

## Shared preprocessing

Categorical columns:

```text
load_area
```

Numeric columns:

```text
hour
day_of_week
month
is_weekend
sin_hour
cos_hour
sin_day_of_week
cos_day_of_week
load_mw
load_lag_1
load_lag_24
load_lag_48
rolling_mean_24
rolling_std_24
```

Use:

```text
sklearn.compose.ColumnTransformer
sklearn.preprocessing.OneHotEncoder
sklearn.preprocessing.StandardScaler
sklearn.pipeline.Pipeline
```

Recommended preprocessing:

```text
OneHotEncoder(handle_unknown="ignore") for load_area
StandardScaler() for numeric columns
```

### `get_preprocessor(categorical_cols, numeric_cols)`

Return a ColumnTransformer.

### `predict_persistence(X: pd.DataFrame) -> np.ndarray`

Persistence baseline:

```text
prediction = X["load_mw"]
```

Because the target is 24 hours ahead, current `load_mw` is the same-hour-yesterday value relative to the target.

No fitting needed.

### `train_ols(X_train, y_train, categorical_cols, numeric_cols)`

Use:

```text
LinearRegression()
```

Return fitted pipeline.

### `train_ridge(X_train, y_train, alpha, categorical_cols, numeric_cols)`

Use:

```text
Ridge(alpha=alpha)
```

Return fitted pipeline.

### `tune_ridge(X_train, y_train, X_val, y_val, alpha_grid, categorical_cols, numeric_cols)`

Use alpha grid:

```text
[0.01, 0.1, 1.0, 10.0, 100.0]
```

Implementation:

```text
for alpha in alpha_grid:
    train ridge on train
    predict on val
    compute validation RMSE
choose alpha with lowest validation RMSE
return best_model, best_alpha, validation_results_df
```

---

# 8. Implement `src/evaluate.py`

## Purpose

Calculate forecast metrics.

## Required functions

### `mae(y_true, y_pred)`

Return mean absolute error.

### `rmse(y_true, y_pred)`

Return root mean squared error.

If using sklearn, note that newer versions may not support `squared=False` consistently. Safe implementation:

```text
sqrt(mean_squared_error(y_true, y_pred))
```

### `mape(y_true, y_pred)`

Return mean absolute percentage error in percent.

Implementation:

```text
mean(abs((y_true - y_pred) / y_true)) * 100
```

Avoid division by zero. If any y_true is zero, use a small epsilon.

### `bias(y_true, y_pred)`

Return:

```text
mean(y_pred - y_true)
```

### `compute_metrics(y_true, y_pred) -> dict`

Return:

```text
{
  "mae": ...,
  "rmse": ...,
  "mape": ...,
  "bias": ...
}
```

### `make_metrics_table(predictions_df: pd.DataFrame) -> pd.DataFrame`

Input long prediction dataframe with:

```text
model
true_load_mw
predicted_load_mw
```

Group by `model` and compute:

```text
mae
rmse
mape
bias
n
```

### `make_metrics_by_load_area(predictions_df: pd.DataFrame) -> pd.DataFrame`

Group by:

```text
model
load_area
```

Compute same metrics.

### `make_error_by_hour(predictions_df: pd.DataFrame) -> pd.DataFrame`

Group by:

```text
model
hour
```

Compute:

```text
mean_abs_error
mean_error
rmse
```

---

# 9. Implement `src/operational.py`

## Purpose

Aggregate load-area predictions into zone-level predictions and compute basic operational metrics.

This is **not** a full dispatch optimizer.

## Required functions

### `aggregate_to_zone(predictions_df: pd.DataFrame) -> pd.DataFrame`

Input long prediction dataframe:

```text
target_timestamp_utc
target_timestamp_ept
model
load_area
true_load_mw
predicted_load_mw
```

Group by:

```text
target_timestamp_utc
target_timestamp_ept
model
```

Compute:

```text
true_zone_load_mw = sum true_load_mw across load areas
predicted_zone_load_mw = sum predicted_load_mw across load areas
n_load_areas = number of unique load areas
```

Important:

```text
Group by target_timestamp_utc, not timestamp_utc.
```

### `add_under_over_generation(zone_df: pd.DataFrame) -> pd.DataFrame`

Add:

```text
zone_error_mw = predicted_zone_load_mw - true_zone_load_mw
zone_abs_error_mw = abs(zone_error_mw)
under_generation_mw = max(0, true_zone_load_mw - predicted_zone_load_mw)
over_generation_mw = max(0, predicted_zone_load_mw - true_zone_load_mw)
```

### `make_operational_metrics(zone_df: pd.DataFrame) -> pd.DataFrame`

Group by `model`.

Compute:

```text
n_hours
under_generation_hours
under_generation_rate
total_under_generation_mw
max_under_generation_mw
total_over_generation_mw
max_over_generation_mw
mean_zone_abs_error_mw
rmse_zone_error_mw
bias_zone_error_mw
```

---

# 10. Implement `src/plots.py`

## Purpose

Create required milestone plots.

Use matplotlib only.

Do not use seaborn.

## Required functions

### `plot_true_vs_predicted_sample(predictions_df, output_path, load_area=None, max_points=96)`

Behavior:

```text
1. If load_area is None, choose the first load area alphabetically.
2. Filter predictions to that load_area.
3. Use first max_points target timestamps in test predictions.
4. Plot true_load_mw and predicted_load_mw for each model.
5. x-axis = target_timestamp_ept.
6. y-axis = MW.
7. Save to output_path.
```

Use separate lines for:

```text
Actual
Persistence
OLS
Ridge
```

### `plot_error_by_hour(error_by_hour_df, output_path)`

Behavior:

```text
1. x-axis = hour
2. y-axis = mean_abs_error
3. one line per model
4. save to output_path
```

### `plot_forecast_metrics_bar(metrics_df, output_path, metric="rmse")`

Optional but useful.

Behavior:

```text
bar chart of selected metric by model
```

---

# 11. Implement `src/run_milestone.py`

## Purpose

Run the entire milestone pipeline from one command.

Command:

```bash
python src/run_milestone.py
```

## Constants

At top of file, set:

```python
RAW_DATA_PATH = "data/hrl_load_metered.csv"
RESULTS_DIR = "results/milestone"
```

Do not require command-line arguments unless desired.

## Pipeline steps

Implement in this exact order:

### Step 1: Create output directories

Create:

```text
results/milestone/metrics
results/milestone/predictions
results/milestone/figures
```

### Step 2: Load and clean data

Call:

```text
load_raw_data
clean_pjm_data
validate_clean_data
```

Optionally save cleaned data to:

```text
results/milestone/predictions/cleaned_data_snapshot.csv
```

### Step 3: Create feature dataset

Call:

```text
make_feature_dataset
```

Optionally save feature dataset to:

```text
results/milestone/predictions/feature_data_snapshot.csv
```

### Step 4: Split data

Call:

```text
time_based_split
validate_split
```

### Step 5: Create X/y/metadata

For train, val, and test, call:

```text
get_feature_target_metadata
```

### Step 6: Train/predict persistence

```text
persistence_test_pred = predict_persistence(X_test)
```

Also compute train/val predictions if desired, but test is required.

### Step 7: Train/predict OLS

```text
ols_model = train_ols(...)
ols_test_pred = ols_model.predict(X_test)
```

### Step 8: Tune/train/predict Ridge

```text
best_ridge_model, best_alpha, ridge_val_results = tune_ridge(...)
ridge_test_pred = best_ridge_model.predict(X_test)
```

Save ridge validation results to:

```text
results/milestone/metrics/ridge_validation_results.csv
```

### Step 9: Build long-format predictions dataframe

Create one dataframe with rows for each model.

Columns:

```text
timestamp_utc
timestamp_ept
target_timestamp_utc
target_timestamp_ept
zone
load_area
model
true_load_mw
predicted_load_mw
error_mw
abs_error_mw
hour
day_of_week
month
```

Models should be named exactly:

```text
Persistence
OLS
Ridge
```

Save to:

```text
results/milestone/predictions/test_predictions.csv
```

### Step 10: Calculate forecast metrics

Create and save:

```text
results/milestone/metrics/forecast_metrics.csv
results/milestone/metrics/forecast_metrics_by_load_area.csv
results/milestone/metrics/error_by_hour.csv
```

### Step 11: Create plots

Save:

```text
results/milestone/figures/true_vs_predicted_sample.png
results/milestone/figures/error_by_hour.png
results/milestone/figures/rmse_by_model.png
```

The third plot is optional but useful.

### Step 12: Zone-level operational metrics

Call:

```text
aggregate_to_zone
add_under_over_generation
make_operational_metrics
```

Save hourly zone predictions to:

```text
results/milestone/predictions/zone_predictions.csv
```

Save metrics to:

```text
results/milestone/metrics/zone_under_over_generation.csv
```

### Step 13: Print final summary

At the end, print:

```text
Data rows after cleaning
Feature rows
Train/val/test rows
Best ridge alpha
Forecast metrics table
Operational metrics table
Paths of saved outputs
```

---

# 12. Expected Outputs

After running:

```bash
python src/run_milestone.py
```

the repo should contain:

```text
results/milestone/metrics/forecast_metrics.csv
results/milestone/metrics/forecast_metrics_by_load_area.csv
results/milestone/metrics/error_by_hour.csv
results/milestone/metrics/ridge_validation_results.csv
results/milestone/metrics/zone_under_over_generation.csv

results/milestone/predictions/test_predictions.csv
results/milestone/predictions/zone_predictions.csv

results/milestone/figures/true_vs_predicted_sample.png
results/milestone/figures/error_by_hour.png
results/milestone/figures/rmse_by_model.png
```

---

# 13. Coding Standards

Use clear, simple, readable Python.

Requirements:

```text
1. Functions should have docstrings.
2. Avoid hardcoding load areas except in validation messages.
3. Use pathlib.Path for paths where possible.
4. Never modify the raw CSV.
5. Save all outputs under results/milestone.
6. Keep the milestone focused; do not implement final-only models.
7. Make the pipeline runnable from repo root with python src/run_milestone.py.
```

---

# 14. Quick Sanity Checks

After implementation, these should be true:

```text
cleaned data rows = 2880
unique timestamps = 720
load areas = 4
feature rows ≈ 2592
models = Persistence, OLS, Ridge
test_predictions.csv has 3 × number_of_test_rows rows
zone_predictions.csv has 3 × number_of_test_timestamps rows
```

If feature rows are not exactly 2592, print why. Small deviations are acceptable if there are missing/duplicate timestamps, but this dataset should likely produce 2592.

---

# 15. Important Model Detail

For this 24-hour-ahead setup:

```text
target_load_mw = load at t + 24
```

Therefore the persistence baseline is:

```text
prediction = load_mw at time t
```

Do not use `load_lag_24` as the main persistence prediction.

`load_lag_24` means load at t - 24, which would be two days before the target.

---

# 16. Do Not Implement These Yet

For milestone-only code, do not implement:

```text
lasso regression
neural network
weekly lag feature set
load_lag_168
rolling_mean_168
dispatch optimizer
reserve-margin experiment
price/LMP data
final report generation
```

Those are final-project items, not milestone items.
