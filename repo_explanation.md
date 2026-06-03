# Energy Demand Forecast With Constraints: Repo Explanation

## Overview

This repository implements a 24-hour-ahead electricity load forecasting pipeline using PJM hourly load data. The current version uses all zones and load areas present in `data/hrl_load_metered.csv`; it no longer filters the dataset down to only the AEP transmission zone.

The project has two evaluation stages:

1. **Pre-constraint layer**: train forecasting models and compare raw load prediction accuracy.
2. **Post-constraint layer**: convert forecasts into generation schedules using a synthetic dispatch constraint layer, then compare operational outcomes such as under-generation, over-generation, and dispatch cost.

The key question is not only "which model predicts load best?" but also "what happens when each forecast is used to make a generation scheduling decision?"

## Data

The input file is:

```text
data/hrl_load_metered.csv
```

The raw file currently contains 90,120 data rows across 7 zones and 11 load-area series.

Current zones:

```text
AEP, AP, ATSI, CE, DEOK, EKPC, OVEC
```

Current zone/load-area series:

```text
AEP / AEPAPT
AEP / AEPIMP
AEP / AEPKPT
AEP / AEPOPT
AP / AP
ATSI / OE
ATSI / PAPWR
CE / CE
DEOK / DEOK
EKPC / EKPC
OVEC / OVEC
```

The cleaned dataframe keeps all zones and standardizes the raw columns into:

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

Cleaning is implemented in `src/data.py`. It parses timestamps, renames columns, converts `mw` to numeric `load_mw`, converts verification values to booleans, strips text columns, drops missing required values, removes duplicate `(timestamp_utc, zone, load_area)` rows, and sorts the data. It does not filter to AEP.

## Feature Engineering

Feature creation is implemented in `src/features.py`.

Each training example predicts the same zone/load-area load 24 hours ahead:

```text
features at time t for zone z and load area a
    -> load at time t + 24 hours for the same zone z and load area a
```

The model features are:

```text
zone
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

The lag and target features are timestamp-exact within each `(zone, load_area)` series. For example, `load_lag_24` is the load exactly 24 hours before the input timestamp, and the target is the load exactly 24 hours after the input timestamp. Rows without complete lag, rolling, or exact target values are dropped.

In the latest run:

```text
Cleaned rows: 90120
Feature rows: 88944
Train/validation/test rows: 62256 / 13341 / 13347
Train/validation/test series: 11 / 11 / 11
```

## Train, Validation, And Test Split

Splitting is implemented in `src/split.py`.

The split is chronological within each `(zone, load_area)` series:

```text
first 70% of each series -> train
next 15% of each series  -> validation
final 15% of each series -> test
```

This matters because not every load area has the same timestamp coverage. A global timestamp split can accidentally exclude shorter series from the test set. The current per-series split keeps all 11 load-area series represented in train, validation, and test while still avoiding future leakage within each series.

## Forecasting Models

Forecasting models are implemented in `src/models.py`.

The current models are:

```text
Persistence
Ordinary Least Squares
Ridge regression
Lasso regression
```

### Persistence

Persistence is the baseline. It predicts that the 24-hour-ahead load equals the current load:

```text
prediction(t + 24) = load_mw(t)
```

It is not trained.

### OLS

OLS is a standard linear regression model. It uses the full feature set after preprocessing:

- `zone` and `load_area` are one-hot encoded.
- Numeric features are standardized.

### Ridge

Ridge is an L2-regularized linear regression model. It uses the same features and preprocessing as OLS, then tunes `alpha` on the validation set using RMSE.

Current best Ridge alpha:

```text
0.01
```

### Lasso

Lasso is an L1-regularized linear regression model. It also uses the same features and preprocessing. The implementation uses `LassoLars`, which is faster for this dataset because there are many rows but relatively few engineered features.

Current best Lasso alpha:

```text
0.001
```

## Pre-Constraint Evaluation

Pre-constraint evaluation is implemented in `src/evaluate.py`.

The pipeline computes:

```text
MAE
RMSE
MAPE
Bias
Error by hour
Metrics by model, zone, and load area
```

Current overall pre-constraint metrics:

| Model | MAE | RMSE | MAPE | Bias |
|---|---:|---:|---:|---:|
| Persistence | 235.36 | 434.43 | 7.69% | -0.55 |
| OLS | 250.26 | 404.01 | 35.00% | 111.26 |
| Ridge | 250.26 | 404.01 | 35.00% | 111.26 |
| Lasso | 238.80 | 395.98 | 34.01% | 63.22 |

In the current run, Lasso has the lowest RMSE, while Persistence has the lowest MAPE and nearly zero bias. OLS and Ridge remain very similar because they are both linear models using the same features, and the selected Ridge regularization is weak.

## Plot Outputs

Plots are generated by `src/plots.py`.

Per-load-area pre-constraint true-vs-predicted plots are now organized in:

```text
results/milestone/figures/pre_constraint_load_area/
```

This folder currently contains 11 plots, one for each zone/load-area series:

```text
pre_constraint_layer_true_vs_predicted_AEP_AEPAPT.png
pre_constraint_layer_true_vs_predicted_AEP_AEPIMP.png
pre_constraint_layer_true_vs_predicted_AEP_AEPKPT.png
pre_constraint_layer_true_vs_predicted_AEP_AEPOPT.png
pre_constraint_layer_true_vs_predicted_AP_AP.png
pre_constraint_layer_true_vs_predicted_ATSI_OE.png
pre_constraint_layer_true_vs_predicted_ATSI_PAPWR.png
pre_constraint_layer_true_vs_predicted_CE_CE.png
pre_constraint_layer_true_vs_predicted_DEOK_DEOK.png
pre_constraint_layer_true_vs_predicted_EKPC_EKPC.png
pre_constraint_layer_true_vs_predicted_OVEC_OVEC.png
```

Each true-vs-predicted plot has two panels:

1. Actual load and model predictions.
2. Signed prediction error, `prediction - actual`.

The error panel helps reveal model differences when OLS, Ridge, and Lasso overlap visually in the main prediction panel.

## Zone Aggregation

Zone aggregation is implemented in `src/operational.py`.

Before dispatch, load-area predictions are aggregated to zone-level predictions:

```text
predicted_zone_load_mw =
    sum(predicted_load_mw for all load areas in the same zone and target timestamp)

true_zone_load_mw =
    sum(true_load_mw for all load areas in the same zone and target timestamp)
```

Grouping is by:

```text
target_timestamp_utc
target_timestamp_ept
zone
model
```

This keeps zones separate. AEP load areas are summed into AEP zone load, ATSI load areas are summed into ATSI zone load, and one-load-area zones remain as one-load-area zone totals.

## Current Constraint Layer

The constraint layer is implemented in `src/constraints.py`.

Its purpose is to convert each model's zone-level forecast into a generation schedule, then compare that schedule to actual zone load.

This layer is not a machine learning model. It is a deterministic post-processing simulation.

### Synthetic Generator Fleet

The repo creates a synthetic generator fleet using:

```python
create_generator_fleet(max_zone_load_mw)
```

The fleet is sized from the maximum true zone load observed in the zone-level prediction input. It has four generator types:

| Generator | Capacity | Cost |
|---|---:|---:|
| cheap_base | 35% of max zone load | $25/MWh |
| mid_cost | 35% of max zone load | $50/MWh |
| high_cost | 35% of max zone load | $85/MWh |
| peaker | 25% of max zone load | $150/MWh |

Total synthetic capacity is:

```text
35% + 35% + 35% + 25% = 130% of max observed zone load
```

The fleet is synthetic. It does not represent real PJM generator units, real fuel costs, real market offers, or real operational constraints.

### Greedy Dispatch

Dispatch is greedy economic dispatch:

```text
1. Dispatch cheap_base first.
2. Then mid_cost.
3. Then high_cost.
4. Then peaker.
```

For a given demand value:

```text
generation_g = min(generator_capacity_g, remaining_demand)
remaining_demand = remaining_demand - generation_g
```

The base generator cost is:

```text
base_generator_cost = sum(generation_mw_g * cost_per_mwh_g)
```

The current implementation vectorizes this calculation so all model-zone-hour forecasts can be dispatched efficiently.

### Penalty Costs And Operational Cost

The upgraded constraint layer now evaluates schedules using one operational cost framework. It keeps the base generator cost, then adds penalties for forecast-driven under-generation and over-generation.

Penalty constants:

```text
under-generation penalty = $1,000/MWh
over-generation penalty  = $50/MWh
```

Hourly penalty formulas:

```text
under_generation_mwh = under_generation_mw
over_generation_mwh  = over_generation_mw

under_generation_penalty_cost = under_generation_mwh * 1000
over_generation_penalty_cost  = over_generation_mwh * 50
penalty_cost = under_generation_penalty_cost + over_generation_penalty_cost
total_operational_cost = base_generator_cost + penalty_cost
```

Over-generation is not charged the full generator cost again. Extra scheduled generation is already reflected in `base_generator_cost`; the over-generation penalty is only an additional balancing, curtailment, or inefficiency penalty.

### Forecast Dispatch Versus Oracle Dispatch

For every model, zone, and target hour, the constraint layer computes two schedules:

1. **Forecast dispatch**: dispatch generation using the model's predicted zone load.
2. **Oracle dispatch**: dispatch generation using the true zone load.

The oracle dispatch is a perfect-information benchmark. It is not a deployable model.

The raw base-cost gap is still available:

```text
cost_gap = base_generator_cost - oracle_base_generator_cost
```

The main regret metric is now:

```text
constraint_regret = total_operational_cost - oracle_total_operational_cost
```

The oracle has zero penalty cost because it dispatches against true load:

```text
oracle_total_operational_cost = oracle_base_generator_cost
```

### Operational Metrics

After dispatching against the forecast, the layer compares scheduled generation to actual zone load:

```text
under_generation_mw = max(0, true_zone_load_mw - scheduled_generation_mw)
over_generation_mw  = max(0, scheduled_generation_mw - true_zone_load_mw)
```

The main post-constraint metrics are:

```text
feasible_hours
infeasible_hours
under_generation_hours
over_generation_hours
under_generation_rate
over_generation_rate
total_under_generation_mwh
total_over_generation_mwh
max_under_generation_mw
max_over_generation_mw
total_base_generator_cost
total_oracle_base_generator_cost
total_under_generation_penalty_cost
total_over_generation_penalty_cost
total_penalty_cost
total_operational_cost
oracle_total_operational_cost
total_constraint_regret
mean_constraint_regret
mean_abs_zone_error_after_dispatch_mw
rmse_zone_error_after_dispatch_mw
bias_zone_error_after_dispatch_mw
```

Current post-constraint metrics:

| Model | Under-Gen Rate | Over-Gen Rate | Under-Gen MWh | Over-Gen MWh | Base Cost | Penalty Cost | Operational Cost | Constraint Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 47.90% | 51.30% | 1,519,251.74 | 1,511,968.31 | 1,641,770,699.20 | 1,594,850,152.70 | 3,236,620,851.90 | 1,595,734,010.65 |
| OLS | 31.00% | 69.00% | 769,260.30 | 2,274,266.31 | 1,706,974,626.45 | 882,973,613.38 | 2,589,948,239.83 | 949,061,398.58 |
| Ridge | 30.98% | 69.02% | 769,272.18 | 2,274,246.33 | 1,706,973,140.88 | 882,984,493.54 | 2,589,957,634.42 | 949,070,793.17 |
| Lasso | 36.66% | 63.34% | 1,000,412.35 | 1,868,103.57 | 1,676,109,421.21 | 1,093,817,530.81 | 2,769,926,952.02 | 1,129,040,110.77 |

### How To Interpret The Constraint Results

The post-constraint layer changes the meaning of evaluation.

Pre-constraint metrics ask:

```text
How close was the forecast to true load?
```

Post-constraint metrics ask:

```text
What operational schedule did the forecast create?
```

This is why a model can look good on one dimension and worse on another. For example:

- A model that underpredicts may have lower base generator cost because it schedules less generation.
- But under-generation is very expensive under the penalty framework.
- A model that overpredicts may reduce under-generation but increase over-generation, base generator cost, and over-generation penalties.

In the current run, OLS has the lowest `total_constraint_regret`, narrowly beating Ridge. Persistence has the lowest base generator cost, but it has the highest under-generation penalty and therefore the highest operational regret. This is the key reason the upgraded constraint layer should rank models by `total_constraint_regret` or `total_operational_cost`, not raw base generator cost alone.

## What The Constraint Layer Does Not Model

The current constraint layer is intentionally simplified. It does not model:

```text
real PJM generator fleets
unit commitment
ramp limits
minimum generation
startup or shutdown costs
transmission constraints
power flow
reserve requirements
renewable uncertainty
fuel constraints
outages
locational marginal prices
market bidding behavior
interchange between zones
```

The only hard operational structure currently modeled is:

```text
forecasted demand must be served by a finite synthetic generator fleet with capacity limits and costs
```

So the dispatch results should be interpreted as scenario-based comparisons under the repo's assumptions, not as real PJM operating-cost estimates.

## Main Output Files

Metrics:

```text
results/milestone/metrics/pre_constraint_layer_forecast_metrics.csv
results/milestone/metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv
results/milestone/metrics/pre_constraint_layer_error_by_hour.csv
results/milestone/metrics/pre_constraint_layer_ridge_validation_results.csv
results/milestone/metrics/pre_constraint_layer_lasso_validation_results.csv
results/milestone/metrics/post_constraint_layer_generator_fleet.csv
results/milestone/metrics/post_constraint_layer_dispatch_metrics.csv
results/milestone/metrics/pre_post_constraint_layer_summary.csv
```

Predictions:

```text
results/milestone/predictions/pre_constraint_layer_cleaned_data_snapshot.csv
results/milestone/predictions/pre_constraint_layer_feature_data_snapshot.csv
results/milestone/predictions/pre_constraint_layer_test_predictions.csv
results/milestone/predictions/pre_constraint_layer_zone_predictions.csv
results/milestone/predictions/post_constraint_layer_dispatch_hourly.csv
```

Figures:

```text
results/milestone/figures/pre_constraint_load_area/
results/milestone/figures/pre_constraint_error/
results/milestone/figures/post_constraint_analysis/
```

## How To Run

From the repository root:

```bash
.venv/bin/python -u src/run.py
```

The `-u` flag makes Python print progress messages immediately, which is useful because the larger multi-zone dataset takes longer than the earlier AEP-only run.
