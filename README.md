# slackshare

Input-oriented Free Disposal Hull (FDH) slack analysis for a single
(typically undesirable) input, e.g. emissions, waste, energy use.

## Concept

For each DMU *i* with input $x_i$ and output vector $y_i$, `slackshare` computes input-oriented FDH (Free Disposal Hull) efficiency and slack:

$$x_i^{*} = \min\{x_j : y_j \ge y_i \text{ on every output dimension}\}$$

$$\text{slack}_i = x_i - x_i^{*}$$

where $x_i^*$ is the smallest input among all DMUs that *dominate* DMU *i* on every output dimension simultaneously (vector or Pareto dominance). The slack $\text{slack}_i$ is the avoidable excess input — how much the DMU could reduce input while maintaining its output performance.

FDH imposes only free disposability (no convexity), so a DMU is only compared against *actually observed* dominating units — never a hypothetical convex blend of several units.

`slackshare` then decomposes total input across all DMUs into aggregate and per-DMU measures:

**Aggregate decomposition:**
$$\text{slack\_share} = \frac{\sum_i \text{slack}_i}{\sum_i x_i}$$

the proportion of total input across all DMUs that is avoidable excess.

**Per-DMU shares:**
- Share of total input: $\dfrac{\text{slack}_i}{\sum_i x_i}$ — DMU's slack as a fraction of the population's total input.
- Share of aggregate slack: $\dfrac{\text{slack}_i}{\sum_i \text{slack}_i}$ — how much of the system-wide waste this DMU accounts for.

## Install

```bash
pip install -e .
```

## Usage

```python
import pandas as pd
import slackshare as ss

data = pd.DataFrame({
    "dmu":       ["A", "B", "C", "D"],
    "emissions": [100, 80, 120, 60],
    "output":    [10, 10, 15, 8],
})

per_dmu, summary = ss.analyze(data, input_col="emissions", output_cols=["output"])

print(per_dmu)
print(summary)
```

Multiple outputs (vector dominance) — just pass a list:

```python
per_dmu, summary = ss.analyze(
    data,
    input_col="emissions",
    output_cols=["revenue", "quality_score"],
)
```

`per_dmu` columns:

| column | meaning |
|---|---|
| `x_star` | FDH-efficient input level |
| `slack` | avoidable excess input (`x - x_star`) |
| `efficient` | True if `slack == 0` |
| `share_of_total_input` | `slack_i / sum(x_i)` |
| `share_of_total_slack` | `slack_i / sum(slack_i)` |

`summary` keys: `total_input`, `total_slack`, `slack_share`.

## Step-by-step API

```python
scored   = ss.fdh_scores(data, input_col="emissions", output_cols=["output"])
summary  = ss.aggregate(scored, input_col="emissions")
with_shr = ss.dmu_shares(scored, input_col="emissions")
```

All functions beyond `fdh_scores` require the `input_col` keyword argument to identify which column holds the input values.

## Panel data

For multi-date, multi-input-type workflows, use `analyze_panel`:

```python
per_dmu_df, summary_df = ss.analyze_panel(input_df, output_df)
```

where `input_df` and `output_df` are long-format DataFrames with columns `(dmu, date, input/output, value)`. The function runs FDH analysis independently across each `(date, input_type)` slice and returns per-DMU results and summary statistics concatenated across all slices. See [`examples/`](examples/) for a complete worked example with synthetic data and interactive Plotly visualizations.

## Development

Run tests with:
```bash
pytest
```

## Notes / scope (v0.1)

- Single (undesirable) input; one or more outputs via vector/Pareto dominance.
- No convexity assumption (this is FDH, not DEA) — so scores reflect pure technical/dominance-based inefficiency, not allocative or convexity-driven inefficiency.
- Panel data support via `analyze_panel` for multi-date, multi-input-type workflows.
