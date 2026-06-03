# Constraint Layer Cost/Regret Upgrade Instructions

## Purpose

Update the post-constraint layer so that model comparison is based on a single operational cost framework rather than separate raw dispatch-cost and under-generation metrics.

The current constraint layer already converts zone-level load forecasts into generation schedules using a synthetic generator fleet. It currently reports dispatch cost and under-generation, but the evaluation should be expanded to explicitly account for both under-generation and over-generation, add penalty costs, rename the raw generation cost to avoid confusion, and compute a total operational regret metric against an oracle dispatch.

The goal is to make the post-constraint comparison answer:

> Given a model forecast, how expensive was the resulting constrained generation schedule after accounting for base generator cost, under-generation penalties, and over-generation penalties?

---

## Required Changes Summary

Implement the following changes in the constraint-layer evaluation:

1. Add a **total over-generation metric**, analogous to the existing total under-generation metric.
2. Add penalty scalars:
   - Under-generation / underprediction penalty: **$10,000/MWh**
   - Over-generation / overprediction penalty: **$50/MWh**
3. Rename the current raw dispatch-cost metric from `total_dispatch_cost` to something clearer, preferably:
   - `total_base_generator_cost`
4. Add a new penalty-cost metric:
   - `total_penalty_cost`
5. Add a new total-cost metric:
   - `total_operational_cost = total_base_generator_cost + total_penalty_cost`
6. Add a regret metric:
   - `total_constraint_regret = total_operational_cost - oracle_total_operational_cost`
7. Ensure all new metrics are:
   - computed at the hourly dispatch-result level where appropriate,
   - aggregated into post-constraint model metrics,
   - saved to CSV,
   - included in the pre/post summary if applicable,
   - visualized with graphs.

---

## Conceptual Definitions

The project should distinguish between the following:

### Base Generator Cost

This is the cost of scheduled generation from the synthetic fleet.

Rename the existing dispatch cost to make clear that it is only the cost of generation scheduled by the model forecast.

Recommended hourly column name:

```python
base_generator_cost
```

Recommended aggregate metric name:

```python
total_base_generator_cost
```

This replaces or aliases the older naming:

```python
dispatch_cost
total_dispatch_cost
forecast_dispatch_cost
```

If keeping backward compatibility is easier, keep the old columns but add the new clearer columns. However, the final output tables and plots should use `base_generator_cost` / `total_base_generator_cost`.

---

### Under-Generation

Under-generation occurs when the model-driven schedule produces less generation than actual load.

Hourly formula:

```python
under_generation_mw = max(0, true_zone_load_mw - scheduled_generation_mw)
```

Because the data is hourly, this can also be treated as MWh:

```python
under_generation_mwh = under_generation_mw
```

Aggregate metric:

```python
total_under_generation_mwh = sum(under_generation_mwh)
```

If the repo currently names this `total_under_generation_mw`, either rename to `total_under_generation_mwh` or add a duplicate MWh column for clarity.

---

### Over-Generation

Over-generation occurs when the model-driven schedule produces more generation than actual load.

Hourly formula:

```python
over_generation_mw = max(0, scheduled_generation_mw - true_zone_load_mw)
```

Because the data is hourly, this can also be treated as MWh:

```python
over_generation_mwh = over_generation_mw
```

Aggregate metric:

```python
total_over_generation_mwh = sum(over_generation_mwh)
```

This should be added as a major headline metric, equivalent in importance to total under-generation.

---

## Penalty Constants

Add constants near the top of the relevant constraint-layer file, likely `src/constraints.py`.

```python
UNDER_GENERATION_PENALTY_PER_MWH = 10_000.0
OVER_GENERATION_PENALTY_PER_MWH = 50.0
```

Use these exact values for the default milestone run.

Interpretation:

- Under-generation is very expensive because it represents failure to serve load, emergency balancing, or reliability risk.
- Over-generation is much less expensive because the base generator cost already accounts for the extra scheduled generation. The $50/MWh over-generation penalty should be interpreted as an additional balancing, curtailment, or inefficiency cost, not the full cost of energy production.

---

## Hourly Cost Calculations

For each model-zone-hour row in the post-constraint hourly dispatch output, compute the following.

### 1. Base Generator Cost

Use the existing forecast-driven dispatch cost, but rename it:

```python
base_generator_cost = forecast_dispatch_cost
```

or, if the current column is simply called `dispatch_cost`:

```python
base_generator_cost = dispatch_cost
```

### 2. Under-Generation Penalty Cost

```python
under_generation_penalty_cost = (
    under_generation_mwh * UNDER_GENERATION_PENALTY_PER_MWH
)
```

### 3. Over-Generation Penalty Cost

```python
over_generation_penalty_cost = (
    over_generation_mwh * OVER_GENERATION_PENALTY_PER_MWH
)
```

### 4. Total Penalty Cost

```python
penalty_cost = (
    under_generation_penalty_cost
    + over_generation_penalty_cost
)
```

### 5. Total Operational Cost

```python
total_operational_cost = (
    base_generator_cost
    + penalty_cost
)
```

---

## Oracle Cost Calculations

The oracle uses actual/true load instead of model-predicted load.

The oracle dispatch should theoretically have:

```python
oracle_under_generation_mwh = 0
oracle_over_generation_mwh = 0
oracle_penalty_cost = 0
```

assuming the synthetic generator fleet has enough capacity to serve true load.

Because the current synthetic fleet is sized to 130% of max observed true zone load, the oracle should normally be feasible.

Compute:

```python
oracle_base_generator_cost = oracle_dispatch_cost
oracle_penalty_cost = 0.0
oracle_total_operational_cost = (
    oracle_base_generator_cost
    + oracle_penalty_cost
)
```

Then compute regret:

```python
constraint_regret = (
    total_operational_cost
    - oracle_total_operational_cost
)
```

Hourly column:

```python
constraint_regret
```

Aggregate model metric:

```python
total_constraint_regret = sum(constraint_regret)
```

Also include:

```python
mean_constraint_regret = mean(constraint_regret)
```

---

## Important Note About Over-Generation Double Counting

Do **not** treat over-generation as if the full energy cost needs to be added again.

The base generator cost already increases when a model overpredicts and schedules extra generation. Therefore:

```python
total_operational_cost = base_generator_cost + penalty_cost
```

where the over-generation penalty is only an **additional** balancing/curtailment/inefficiency penalty.

Do **not** calculate:

```python
base_generator_cost
+ over_generation_mwh * full_generator_cost_again
```

because that double-counts the cost of over-scheduled generation.

---

## Required Hourly Output Columns

Update the hourly post-constraint dispatch output, likely:

```text
results/milestone/predictions/post_constraint_layer_dispatch_hourly.csv
```

It should include at least the following columns:

```text
target_timestamp_utc
target_timestamp_ept
zone
model
true_zone_load_mw
predicted_zone_load_mw
scheduled_generation_mw
base_generator_cost
oracle_base_generator_cost
under_generation_mw
over_generation_mw
under_generation_mwh
over_generation_mwh
under_generation_penalty_cost
over_generation_penalty_cost
penalty_cost
total_operational_cost
oracle_total_operational_cost
constraint_regret
```

If the current implementation already has similarly named columns, preserve them if useful, but add these standardized columns.

---

## Required Aggregated Metrics

Update the post-constraint metrics file, likely:

```text
results/milestone/metrics/post_constraint_layer_dispatch_metrics.csv
```

The model-level metrics should include at least:

```text
model
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

Recommended formulas:

```python
under_generation_hours = (under_generation_mwh > 0).sum()
over_generation_hours = (over_generation_mwh > 0).sum()

under_generation_rate = under_generation_hours / total_hours
over_generation_rate = over_generation_hours / total_hours

total_under_generation_mwh = under_generation_mwh.sum()
total_over_generation_mwh = over_generation_mwh.sum()

total_base_generator_cost = base_generator_cost.sum()
total_oracle_base_generator_cost = oracle_base_generator_cost.sum()

total_under_generation_penalty_cost = under_generation_penalty_cost.sum()
total_over_generation_penalty_cost = over_generation_penalty_cost.sum()
total_penalty_cost = penalty_cost.sum()

total_operational_cost = total_operational_cost.sum()
oracle_total_operational_cost = oracle_total_operational_cost.sum()

total_constraint_regret = constraint_regret.sum()
mean_constraint_regret = constraint_regret.mean()
```

---

## Required Summary Output

Update the pre/post summary file if it exists:

```text
results/milestone/metrics/pre_post_constraint_layer_summary.csv
```

Include the new post-constraint metrics so the final model comparison includes:

```text
pre_constraint_rmse
pre_constraint_mae
pre_constraint_mape
pre_constraint_bias
total_under_generation_mwh
total_over_generation_mwh
total_base_generator_cost
total_penalty_cost
total_operational_cost
total_constraint_regret
```

The final comparison should make it easy to see whether the model with the best forecasting error is also the model with the best operational cost/regret.

---

## Required Graphs

Add graphs for the new metrics. Use the existing plotting style and output folder convention.

Expected output folder:

```text
results/milestone/figures/
```

Add at least the following figures:

### 1. Total Over-Generation By Model

Filename:

```text
post_constraint_layer_over_generation_by_model.png
```

Y-axis:

```text
Total over-generation (MWh)
```

X-axis:

```text
Model
```

Metric:

```python
total_over_generation_mwh
```

---

### 2. Base Generator Cost By Model

Filename:

```text
post_constraint_layer_base_generator_cost_by_model.png
```

Y-axis:

```text
Total base generator cost ($)
```

X-axis:

```text
Model
```

Metric:

```python
total_base_generator_cost
```

This replaces the interpretation of the old dispatch-cost graph.

---

### 3. Penalty Cost By Model

Filename:

```text
post_constraint_layer_penalty_cost_by_model.png
```

Y-axis:

```text
Total penalty cost ($)
```

X-axis:

```text
Model
```

Metric:

```python
total_penalty_cost
```

Optional stacked version:

```text
post_constraint_layer_penalty_cost_stacked_by_model.png
```

Stack components:

```text
total_under_generation_penalty_cost
total_over_generation_penalty_cost
```

---

### 4. Total Operational Cost By Model

Filename:

```text
post_constraint_layer_total_operational_cost_by_model.png
```

Y-axis:

```text
Total operational cost ($)
```

X-axis:

```text
Model
```

Metric:

```python
total_operational_cost
```

---

### 5. Constraint Regret By Model

Filename:

```text
post_constraint_layer_constraint_regret_by_model.png
```

Y-axis:

```text
Total constraint regret ($)
```

X-axis:

```text
Model
```

Metric:

```python
total_constraint_regret
```

This should become one of the main post-constraint result figures.

---

## Recommended Implementation Locations

Likely files to modify:

```text
src/constraints.py
src/operational.py
src/plots.py
src/run_milestone.py
```

Suggested division of work:

### `src/constraints.py`

Add:

```python
UNDER_GENERATION_PENALTY_PER_MWH = 10_000.0
OVER_GENERATION_PENALTY_PER_MWH = 50.0
```

Add hourly cost columns:

```python
under_generation_mwh
over_generation_mwh
under_generation_penalty_cost
over_generation_penalty_cost
penalty_cost
base_generator_cost
oracle_base_generator_cost
total_operational_cost
oracle_total_operational_cost
constraint_regret
```

Update aggregation logic to include all new metrics.

### `src/operational.py`

If this file aggregates zone predictions or calls the constraint layer, make sure the added columns survive through the pipeline and are saved.

### `src/plots.py`

Add plotting functions for:

```text
over-generation by model
base generator cost by model
penalty cost by model
total operational cost by model
constraint regret by model
```

### `src/run_milestone.py`

Make sure the milestone run calls the updated constraint-layer and plotting functions.

---

## Validation Checks

After implementation, verify the following:

### 1. Oracle Penalties Should Be Zero

Check that:

```python
oracle_penalty_cost == 0
```

or approximately zero for all oracle rows.

If not, determine whether this is due to generator-capacity limits or a calculation bug.

### 2. Total Cost Identity

For every hourly row:

```python
total_operational_cost == base_generator_cost + penalty_cost
```

For every model aggregate:

```python
total_operational_cost == total_base_generator_cost + total_penalty_cost
```

### 3. Regret Identity

For every hourly row:

```python
constraint_regret == total_operational_cost - oracle_total_operational_cost
```

For every model aggregate:

```python
total_constraint_regret == total_operational_cost - oracle_total_operational_cost
```

### 4. Under/Over Generation Cannot Both Be Positive

For every hourly row, this should hold:

```python
not (under_generation_mwh > 0 and over_generation_mwh > 0)
```

A row should not be both under-generated and over-generated at the same time.

### 5. No Negative Penalty Costs

All of the following should be nonnegative:

```python
under_generation_penalty_cost
over_generation_penalty_cost
penalty_cost
```

### 6. Over-Generation Now Appears In Final Metrics

The final CSV and plots should make over-generation as visible as under-generation.

---

## Example Expected Interpretation

After the update, the project should no longer compare models using raw dispatch cost alone.

Raw dispatch cost can make underpredicting models look artificially good because they schedule too little generation.

The preferred final comparison should use:

```text
total_constraint_regret
```

or:

```text
total_operational_cost
```

The interpretation should be:

- `total_base_generator_cost` tells how expensive the scheduled generation was before penalties.
- `total_under_generation_mwh` tells how much load was not served by the model-driven schedule.
- `total_over_generation_mwh` tells how much extra generation was scheduled.
- `total_penalty_cost` monetizes under-generation and over-generation errors.
- `total_operational_cost` combines generation cost and penalty cost.
- `total_constraint_regret` compares the model-driven schedule to an oracle schedule using actual load.

The best model under the constraint layer is the model with the lowest `total_constraint_regret`, not necessarily the model with the lowest raw RMSE or the lowest base generator cost.

---

## Acceptance Criteria

The task is complete when:

1. `post_constraint_layer_dispatch_hourly.csv` contains hourly over-generation, penalty-cost, total-cost, and regret columns.
2. `post_constraint_layer_dispatch_metrics.csv` contains model-level aggregate metrics for:
   - total over-generation,
   - base generator cost,
   - under-generation penalty cost,
   - over-generation penalty cost,
   - total penalty cost,
   - total operational cost,
   - total constraint regret.
3. The old `total_dispatch_cost` interpretation is replaced with `total_base_generator_cost`.
4. Graphs are created for:
   - total over-generation,
   - base generator cost,
   - penalty cost,
   - total operational cost,
   - constraint regret.
5. Oracle dispatch has zero or near-zero under/over-generation penalty.
6. The final reportable model ranking can be based on `total_constraint_regret`.
