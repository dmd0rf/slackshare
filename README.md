# slackshare

Input-oriented Free Disposal Hull (FDH) slack analysis for a single (typically undesirable) input, e.g. emissions, waste, energy use.

## Visual Overview

**Slack by DMU:** Each row is a decision-making unit (DMU). Color intensity represents avoidable input slack (purple = efficient, yellow = wasteful). Switch between 40 input types via the dropdown. Interactive: hover to see exact values and the benchmark peer DMU.

![Slack by DMU Heatmap](demo1.png)

**Aggregate Trends:** Bars show total input consumption per month. The orange line shows slack share — the percentage of total input that is avoidable waste. This reveals whether your population is becoming more or less efficient over time.

![Total Input & Slackshare Trend](demo2.png)

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
| `peer` | identifier of the DMU that achieves `x_star` for this DMU; among ties, alphabetically first |

`summary` keys: `total_input`, `total_slack`, `slack_share`.

## Step-by-step API

```python
scored   = ss.fdh_scores(data, input_col="emissions", output_cols=["output"], id_col="dmu")
summary  = ss.aggregate(scored, input_col="emissions")
with_shr = ss.dmu_shares(scored, input_col="emissions")
```

All functions beyond `fdh_scores` require the `input_col` keyword argument to identify which column holds the input values.

Optional `id_col` (for `fdh_scores` and `analyze`) specifies which column contains DMU identifiers for peer identification; if omitted, uses DataFrame index.

## Panel Data

For multi-date, multi-input-type workflows, use `analyze_panel`:

```python
per_dmu_df, summary_df = ss.analyze_panel(input_df, output_df)
```

where `input_df` and `output_df` are long-format DataFrames with columns `(dmu, date, input/output, value)`. The function runs FDH analysis independently across each `(date, input_type)` slice and returns per-DMU results and summary statistics concatenated across all slices.

## Interactive Example

Run a complete end-to-end example that generates synthetic data, runs FDH analysis, and builds the two interactive HTML charts you see above.

### Setup

Install the optional dependencies:

```bash
pip install -e ".[example]"
```

### Run

```bash
python run_example.py
```

This generates a synthetic panel (500 DMUs, 40 input types, 12 output types across 12 monthly dates), runs FDH analysis, and builds two interactive HTML reports. Total runtime: ~10 seconds.

Outputs: `output/slack_by_dmu.html` and `output/slack_share.html`

### View the reports

Open the HTML files in your browser:

```bash
open output/slack_by_dmu.html      # macOS
xdg-open output/slack_by_dmu.html  # Linux
start output\slack_by_dmu.html     # Windows
```

The HTML files are fully self-contained and work offline.

### What you'll see

**output/slack_by_dmu.html** — Interactive heatmap showing slack (waste) for each DMU over time.
- X-axis: 12 monthly dates
- Y-axis: 500 DMUs (sorted by slack)
- Color: intensity of slack (purple = efficient, yellow = wasteful)
- Dropdown: switch between the 40 input types
- Hover: see exact DMU id, date, slack value, and the peer DMU that set the benchmark

**output/slack_share.html** — System-wide trend chart.
- Bars: total input consumption per date
- Line: slack share (% of input that is avoidable waste)
- Dropdown: switch between input types
- Hover: see exact values

### Scale up

To run on a larger dataset (slower, O(n²) runtime):

```bash
python run_example.py --n-dmus 2000  # ~30s
python run_example.py --n-dmus 5000  # ~3 minutes
```

## Development

Run tests with:
```bash
pytest
```

## Notes / scope (v0.1)

- Single (undesirable) input; one or more outputs via vector/Pareto dominance.
- No convexity assumption (this is FDH, not DEA) — so scores reflect pure technical/dominance-based inefficiency, not allocative or convexity-driven inefficiency.
- Panel data support via `analyze_panel` for multi-date, multi-input-type workflows.

## See also

- [paper](docs/paper.pdf) — Full formalization with worked examples and theoretical analysis.
