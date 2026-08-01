# slackshare

Input-oriented Free Disposal Hull (FDH) slack analysis for a single
(typically undesirable) input, e.g. emissions, waste, energy use.

## Concept

For each DMU *i* with input `x_i` and output `y_i`, `slackshare` finds the
FDH-efficient input level:

```
x_i*    = min{ x_j : y_j >= y_i }     # smallest input among units that
                                       # produce at least as much output
slack_i = x_i - x_i*                  # avoidable excess input
```

Outputs can be a single column or several. With multiple outputs, `y_j >= y_i`
means **vector (Pareto) dominance**: unit *j* must match or exceed unit *i*
on **every** output dimension to count as a dominator.

FDH imposes only free disposability (no convexity), so a DMU is only
compared against *actually observed* dominating units — never a
hypothetical convex blend of several units.

`slackshare` then decomposes total input across all DMUs into:

- **Aggregate slack share** — the proportion of total input across all
  DMUs that is avoidable excess:
  `slack_share = sum(slack_i) / sum(x_i)`
- **Each DMU's share of total input** attributable to its own slack:
  `slack_i / sum(x_i)`
- **Each DMU's share of aggregate slack** — how much of the *system-wide
  waste* a given DMU accounts for:
  `slack_i / sum(slack_i)`

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
summary  = ss.aggregate(scored)
with_shr = ss.dmu_shares(scored)
```

## Notes / scope (v0.1)

- Single (undesirable) input; one or more outputs via vector/Pareto
  dominance.
- No convexity assumption (this is FDH, not DEA) — so scores reflect
  pure technical/dominance-based inefficiency, not allocative or
  convexity-driven inefficiency.
