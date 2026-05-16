# Implementation Plan: Add Constraint Layer to Existing Milestone Repo

## Purpose

These instructions are for modifying the **current working milestone repo**.

Assumption:

```text
The milestone repo already works.
It already loads the PJM CSV, creates features, trains Persistence/OLS/Ridge, and generates pre-constraint forecast results.
```

Do **not** rebuild the entire project from scratch.

Instead, add a **constraint/dispatch layer** on top of the existing milestone pipeline.

The goal is to compare the milestone models in two stages:

```text
1. pre_constraint_layer:
      compare raw forecasting accuracy before dispatch constraints

2. post_constraint_layer:
      apply constrained dispatch to each model's forecast, then compare operational outcomes
```

The labels `pre_constraint_layer` and `post_constraint_layer` should be used clearly in filenames, result tables, and any plot titles where relevant.

---

# 1. Current Milestone Repo Status

The current milestone repo already has models:

```text
Persistence
OLS
Ridge
```

It already produces pre-constraint forecast results such as:

```text
results/milestone/metrics/pre_constraint_layer_forecast_metrics.csv
results/milestone/metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv
results/milestone/metrics/pre_constraint_layer_error_by_hour.csv
results/milestone/predictions/pre_constraint_layer_test_predictions.csv
results/milestone/figures/pre_constraint_layer_true_vs_predicted_average_load_area.png
results/milestone/figures/pre_constraint_layer_error_by_hour.png
```

These should now be renamed or additionally saved with the prefix:

```text
pre_constraint_layer_
```

For example:

```text
results/milestone/metrics/pre_constraint_layer_forecast_metrics.csv
results/milestone/metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv
results/milestone/metrics/pre_constraint_layer_error_by_hour.csv
results/milestone/predictions/pre_constraint_layer_test_predictions.csv
results/milestone/figures/pre_constraint_layer_true_vs_predicted_average_load_area.png
results/milestone/figures/pre_constraint_layer_error_by_hour.png
```

If the old files are still saved too, that is fine, but the new clearly labeled files must exist.

---

# 2. Conceptual Pipeline After the Change

The updated milestone pipeline should be:

```text
Raw PJM CSV
    ↓
Clean data
    ↓
Create features and 24-hour-ahead target
    ↓
Train Persistence / OLS / Ridge
    ↓
PRE-CONSTRAINT LAYER COMPARISON
    - MAE
    - RMSE
    - MAPE
    - Bias
    - error by hour
    - true vs predicted
    ↓
Aggregate load-area predictions to AEP zone load
    ↓
Apply constrained dispatch layer to each model
    ↓
POST-CONSTRAINT LAYER COMPARISON
    - under-generation
    - over-generation
    - dispatch cost
    - oracle dispatch cost
    - cost gap
    - feasibility
```

The core research structure is:

```text
Compare models before the constraint layer.
Then pass every model through the same hard-constraint dispatch layer.
Then compare models again after the constraint layer.
```

---

# 3. Important Clarification: Are There 6 Total Comparisons?

There are **three models** and **two comparison stages**:

```text
3 models × 2 stages = 6 model-stage evaluations
```

The stages are:

```text
Persistence pre_constraint_layer
OLS pre_constraint_layer
Ridge pre_constraint_layer

Persistence post_constraint_layer
OLS post_constraint_layer
Ridge post_constraint_layer
```

However, the **metrics are not identical before and after the constraint layer**.

## Pre-constraint metrics

These evaluate pure forecasting performance.

Use:

```text
MAE
RMSE
MAPE
Bias
Error by hour
True vs predicted load plot
```

These compare:

```text
predicted_load_mw vs true_load_mw
```

at the load-area level.

## Post-constraint metrics

These evaluate operational performance after dispatch constraints.

Use:

```text
under_generation_mw
over_generation_mw
dispatch_cost
oracle_dispatch_cost
cost_gap
feasible_for_forecast
```

These compare:

```text
scheduled_generation_mw vs true_zone_load_mw
```

at the zone level.

## Should RMSE/error-by-hour/true-vs-predicted also be computed post-constraint?

Do **not** duplicate all the same load-area forecast plots post-constraint.

Reason:

```text
RMSE, MAPE, and true-vs-predicted are forecasting metrics.
They belong primarily to the pre_constraint_layer.
```

After the constraint layer, the output is not a raw load-area forecast anymore. It is a constrained dispatch/scheduled generation decision at the zone level.

For post-constraint, use operational metrics instead.

Optional post-constraint analogs are allowed:

```text
post_constraint_layer_zone_rmse.csv
post_constraint_layer_zone_error_by_hour.csv
post_constraint_layer_scheduled_vs_true_zone_load.png
```

But the required post-constraint comparison should focus on:

```text
under-generation
over-generation
dispatch cost
cost gap vs oracle
```

So the correct interpretation is:

```text
There are 6 model-stage evaluations, but not 6 copies of the exact same metric set.
```

---

# 4. Add or Modify Module: `src/constraints.py`

Create a new file:

```text
src/constraints.py
```

This file should implement the hard-constraint dispatch layer.

If the repo already has `src/operational.py`, keep it for aggregation if it already works. Use `constraints.py` for generator fleet and constrained dispatch.

---

# 5. Required Functions in `src/constraints.py`

## 5.1 `create_generator_fleet(max_zone_load_mw: float) -> pd.DataFrame`

Create a synthetic generator fleet based on max observed true zone load.

Use this fleet:

```text
Generator       Max MW                   Cost $/MWh
----------------------------------------------------
cheap_base      0.35 * max_zone_load      25
mid_cost        0.35 * max_zone_load      50
high_cost       0.35 * max_zone_load      85
peaker          0.25 * max_zone_load      150
```

Total capacity:

```text
1.30 * max_zone_load
```

This ensures dispatch is feasible while still forcing expensive generators during high demand.

Return dataframe columns:

```text
generator
max_mw
cost_per_mwh
dispatch_order
```

Save this fleet later to:

```text
results/milestone/metrics/post_constraint_layer_generator_fleet.csv
```

---

## 5.2 `greedy_dispatch(demand_mw: float, generator_fleet: pd.DataFrame) -> dict`

Dispatch generators from cheapest to most expensive.

Algorithm:

```text
1. Sort generator_fleet by cost_per_mwh ascending.
2. remaining_demand = demand_mw.
3. For each generator:
      generation_mw = min(generator.max_mw, remaining_demand)
      remaining_demand -= generation_mw
      cost += generation_mw * cost_per_mwh
4. If remaining_demand <= tolerance:
      feasible = True
   else:
      feasible = False
5. Return dictionary with:
      demand_mw
      total_generation_mw
      dispatch_cost
      feasible
      unmet_demand_mw
      generation for each generator
```

Use tolerance:

```text
1e-6
```

Hard constraints enforced:

```text
0 <= generation_mw <= generator.max_mw
total_generation_mw >= demand_mw if feasible
```

This is the actual hard-constraint part of the milestone extension.

---

## 5.3 `run_constrained_dispatch(zone_predictions_df: pd.DataFrame, generator_fleet: pd.DataFrame) -> pd.DataFrame`

Input dataframe should contain:

```text
target_timestamp_utc
target_timestamp_ept
model
true_zone_load_mw
predicted_zone_load_mw
```

For every row:

```text
1. Dispatch predicted_zone_load_mw.
2. Dispatch true_zone_load_mw as oracle.
3. Compute post-constraint metrics.
```

For each row, output:

```text
target_timestamp_utc
target_timestamp_ept
model
true_zone_load_mw
predicted_zone_load_mw
scheduled_generation_mw
dispatch_cost
oracle_generation_mw
oracle_dispatch_cost
cost_gap
feasible_for_forecast
unmet_forecast_demand_mw
under_generation_mw
over_generation_mw
```

Definitions:

```text
scheduled_generation_mw = total generation scheduled to meet predicted_zone_load_mw

oracle_generation_mw = total generation scheduled to meet true_zone_load_mw

cost_gap = dispatch_cost - oracle_dispatch_cost

under_generation_mw = max(0, true_zone_load_mw - scheduled_generation_mw)

over_generation_mw = max(0, scheduled_generation_mw - true_zone_load_mw)
```

Important:

```text
The dispatch layer guarantees scheduled_generation_mw >= predicted_zone_load_mw if feasible.
But if predicted_zone_load_mw < true_zone_load_mw, there can still be under_generation_mw relative to true demand.
```

---

## 5.4 `make_post_constraint_metrics(dispatch_df: pd.DataFrame) -> pd.DataFrame`

Group by `model`.

Compute:

```text
n_hours
feasible_hours
infeasible_hours
under_generation_hours
under_generation_rate
total_under_generation_mw
max_under_generation_mw
total_over_generation_mw
max_over_generation_mw
total_dispatch_cost
total_oracle_dispatch_cost
total_cost_gap
mean_cost_gap
mean_abs_zone_error_after_dispatch_mw
rmse_zone_error_after_dispatch_mw
bias_zone_error_after_dispatch_mw
```

For zone error after dispatch:

```text
zone_error_after_dispatch_mw = scheduled_generation_mw - true_zone_load_mw
```

This gives an optional post-constraint analog to forecast error, but the main focus remains operational metrics.

Save to:

```text
results/milestone/metrics/post_constraint_layer_dispatch_metrics.csv
```

---

# 6. Update Existing Aggregation Logic

The repo likely already aggregates predictions to zone-level.

If not, implement this in `src/operational.py` or similar.

## Required function: `aggregate_predictions_to_zone(predictions_df)`

Input:

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
true_zone_load_mw = sum true_load_mw
predicted_zone_load_mw = sum predicted_load_mw
n_load_areas = nunique load_area
```

Save to:

```text
results/milestone/predictions/pre_constraint_layer_zone_predictions.csv
```

or:

```text
results/milestone/predictions/post_constraint_layer_zone_input_predictions.csv
```

Recommended:

```text
results/milestone/predictions/pre_constraint_layer_zone_predictions.csv
```

because these are still model forecasts before dispatch.

---

# 7. Update `src/plots.py`

Add post-constraint plots.

## 7.1 `plot_post_constraint_dispatch_cost(post_metrics_df, output_path)`

Bar chart:

```text
x-axis = model
y-axis = total_dispatch_cost
title = "Post-Constraint Layer: Total Dispatch Cost by Model"
```

Save to:

```text
results/milestone/figures/post_constraint_layer_dispatch_cost_by_model.png
```

## 7.2 `plot_post_constraint_under_generation(post_metrics_df, output_path)`

Bar chart:

```text
x-axis = model
y-axis = total_under_generation_mw
title = "Post-Constraint Layer: Total Under-Generation by Model"
```

Save to:

```text
results/milestone/figures/post_constraint_layer_under_generation_by_model.png
```

## 7.3 Optional: `plot_post_constraint_scheduled_vs_true(dispatch_df, output_path)`

Line plot over time for each model or selected model:

```text
true_zone_load_mw
scheduled_generation_mw
```

Save to:

```text
results/milestone/figures/post_constraint_layer_scheduled_vs_true_zone_load.png
```

This is optional.

Do not clutter the milestone with too many plots.

---

# 8. Update `src/run_milestone.py`

Modify the existing milestone runner.

Do not remove the current forecasting pipeline.

Add the new constraint-layer steps at the end.

## Required updated pipeline

```text
1. Existing: load raw data.
2. Existing: clean data.
3. Existing: create features.
4. Existing: split data.
5. Existing: train Persistence, OLS, Ridge.
6. Existing: generate model predictions.
7. Existing: save pre-constraint forecast metrics and plots.
8. New: aggregate test predictions to zone level.
9. New: create synthetic generator fleet from max true zone load.
10. New: run constrained dispatch for each model and each test target hour.
11. New: compute post-constraint dispatch metrics.
12. New: save post-constraint hourly dispatch results.
13. New: save post-constraint summary metrics.
14. New: save post-constraint plots.
15. Print both pre-constraint and post-constraint summaries.
```

## Required naming changes

Where possible, update existing outputs to use:

```text
pre_constraint_layer_
```

for pure forecasting results.

For example:

```text
pre_constraint_layer_forecast_metrics.csv
```

should also be saved as:

```text
pre_constraint_layer_forecast_metrics.csv
```

Similarly:

```text
pre_constraint_layer_test_predictions.csv
```

should also be saved as:

```text
pre_constraint_layer_test_predictions.csv
```

Then all dispatch outputs should use:

```text
post_constraint_layer_
```

---

# 9. Required Output Files After Update

After running:

```bash
python3 src/run_milestone.py
```

the repo should create these files.

## 9.1 Pre-constraint layer outputs

```text
results/milestone/metrics/pre_constraint_layer_forecast_metrics.csv
results/milestone/metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv
results/milestone/metrics/pre_constraint_layer_error_by_hour.csv

results/milestone/predictions/pre_constraint_layer_test_predictions.csv
results/milestone/predictions/pre_constraint_layer_zone_predictions.csv

results/milestone/figures/pre_constraint_layer_true_vs_predicted_average_load_area.png
results/milestone/figures/pre_constraint_layer_error_by_hour.png
```

These answer:

```text
Which model predicts load best before constraints?
```

## 9.2 Post-constraint layer outputs

```text
results/milestone/metrics/post_constraint_layer_generator_fleet.csv
results/milestone/metrics/post_constraint_layer_dispatch_metrics.csv

results/milestone/predictions/post_constraint_layer_dispatch_hourly.csv

results/milestone/figures/post_constraint_layer_dispatch_cost_by_model.png
results/milestone/figures/post_constraint_layer_under_generation_by_model.png
```

Optional:

```text
results/milestone/figures/post_constraint_layer_scheduled_vs_true_zone_load.png
```

These answer:

```text
Which model performs best after forecasts are used in a hard-constrained dispatch system?
```

---

# 10. Pre/Post Comparison Tables

Create a combined summary table if easy.

Save to:

```text
results/milestone/metrics/pre_post_constraint_layer_summary.csv
```

Recommended columns:

```text
model
pre_constraint_layer_rmse
pre_constraint_layer_mae
pre_constraint_layer_mape
pre_constraint_layer_bias
post_constraint_layer_under_generation_rate
post_constraint_layer_total_under_generation_mw
post_constraint_layer_total_over_generation_mw
post_constraint_layer_total_dispatch_cost
post_constraint_layer_total_cost_gap
```

This table makes it easy to compare whether the best pre-constraint model is also the best post-constraint model.

---

# 11. What Not to Do

Do not add final-only scope yet.

Do not implement:

```text
Lasso
Neural network
Weekly lag features
Reserve margin experiment
Real LMP/cost data
Complex linear programming dispatch
```

For the milestone extension, the simple greedy dispatch is enough.

---

# 12. Validation Checks

After implementation, verify:

```text
1. Pre-constraint forecast metrics still match old milestone results or are very close.
2. pre_constraint_layer_test_predictions.csv contains Persistence, OLS, Ridge.
3. pre_constraint_layer_zone_predictions.csv has 3 × number_of_test_target_hours rows.
4. post_constraint_layer_generator_fleet.csv has 4 generators.
5. Each generator dispatch obeys 0 <= generation <= max_mw.
6. scheduled_generation_mw >= predicted_zone_load_mw whenever feasible_for_forecast is True.
7. oracle dispatch is feasible for all true_zone_load_mw rows.
8. post_constraint_layer_dispatch_metrics.csv has one row each for Persistence, OLS, Ridge.
```

---

# 13. How to Interpret Results

The milestone now has two separate comparisons.

## Pre-constraint layer interpretation

Use forecast metrics:

```text
MAE
RMSE
MAPE
Bias
```

Question:

```text
Which model predicts load most accurately?
```

## Post-constraint layer interpretation

Use operational metrics:

```text
under_generation
over_generation
dispatch cost
cost gap vs oracle
```

Question:

```text
If each model's forecast is used to schedule generation under hard capacity constraints, which model creates the best operational outcome?
```

Possible insight:

```text
The model with the lowest RMSE may not necessarily have the lowest under-generation or best dispatch cost.
```

That is the main reason for doing both pre- and post-constraint comparisons.
