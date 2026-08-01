# Large-Panel FDH Slack Example

This directory contains a complete example of `slackshare`'s FDH slack analysis on a large synthetic panel: **20,000 DMUs, 40 input types, 12 output types (sparse)**, across **12 monthly dates**.

The example generates synthetic data, runs the analysis, and produces two interactive HTML charts for exploration.

## Setup

Install the package and example dependencies:

```bash
pip install -e ..
pip install -r requirements.txt
```

## Quick Start

### 1. Benchmark first (optional)

Before committing to the full 20,000-DMU analysis, run the benchmark on 500 DMUs to estimate total runtime:

```bash
python run_panel_analysis.py --benchmark
```

This takes seconds–minutes and shows the extrapolated time for the full run:

```
BENCHMARK MODE (500 DMUs)
Expected runtime scales as O(n²), so time(20000) ≈ time(500) × 1600.

Generating panel: n_dmus=500
  Input rows: 240,000
  Output rows: 72,000
Running analyze_panel (this is O(n²) and may take time)...
✓ Analysis complete in 5.2s
  Per-DMU rows: 240,000
  Summary rows: 480

Benchmark time: 5.2s
Estimated time for n_dmus=20000: ~138.6 minutes (8314s)
```

If the extrapolation is > 1 hour, consider using `run_panel_analysis_parallel.py` (see "Parallel execution" below). Otherwise, proceed to step 2.

### 2. Run full analysis

```bash
python run_panel_analysis.py
```

This generates the 20,000-DMU dataset, runs `slackshare.panel.analyze_panel()` across all 480 `(date, input)` slices, and caches the results to parquet:

```
Generating panel: n_dmus=20000
  Input rows: 9,600,000
  Output rows: 2,880,000
Running analyze_panel (this is O(n²) and may take time)...
✓ Analysis complete in 8234.5s
  Per-DMU rows: 9,600,000
  Summary rows: 480
Writing cache to examples/output/cache
```

The cache is written to `output/cache/per_dmu_all_n20000.parquet` and `output/cache/summary_all_n20000.parquet`. Subsequent chart builds reuse this cache without re-running the analysis.

### 3. Build the interactive HTML charts

```bash
python build_html_report.py
```

This reads the cached parquet and builds two Plotly-based HTML reports:

```
Reading cache from examples/output/cache
Per-DMU rows: 9,600,000
Summary rows: 480
Building heatmap report...
✓ Heatmap report: examples/output/slack_by_dmu.html (63.0 MB)
Building bar chart report...
✓ Bar chart report: examples/output/slack_share.html (0.1 MB)

✓ Both reports generated in examples/output
```

### 4. View the reports

Open each HTML file in a web browser. Both files are standalone (no external files needed) but **require an internet connection** to render, since they load Plotly.js from a CDN:

```bash
open output/slack_by_dmu.html      # macOS
xdg-open output/slack_by_dmu.html  # Linux
start output\slack_by_dmu.html     # Windows
```

## What the charts show

### `slack_by_dmu.html` — Slack Heatmap

- **X-axis:** 12 monthly dates (`2024-01` through `2024-12`).
- **Y-axis:** All 20,000 DMUs, sorted by descending mean $|\text{slack}|$ for the currently selected input (order updates when you switch inputs).
- **Cell color:** Absolute value of FDH slack for each DMU/date pair, using a viridis scale (purple = low slack, yellow = high slack).
- **Interaction:**
  - Dropdown to select an input (all 40 available).
  - Scroll to explore all 20,000 DMU rows (one row per DMU).
  - Hover over a cell to see the exact DMU id, date, and slack value.
  - Switching inputs re-sorts rows by that input's slack and recomputes the color scale.

### `slack_share.html` — Total Input & Slack Share

- **X-axis:** 12 monthly dates.
- **Left Y-axis (bars):** Total input value summed across all DMUs for that date.
- **Right Y-axis (line):** Slack share (%) — the percentage of total input that is avoidable excess.
- **Interaction:**
  - Dropdown (shared with the heatmap) to select an input.
  - Hover to see exact values.
  - Charts update instantly when you change the input.

## Data generation details

The synthetic data is designed to be realistic:

- **40 inputs, dense:** every DMU has a value for every input at every date. Inputs fluctuate over time (seasonal + trend + noise).
- **12 outputs, sparse:** each DMU produces only a median of **3** out of 12 possible output types (fixed per DMU across all dates). Inactive outputs are exactly 0.0. The sparsity pattern ensures that FDH dominance is not trivial — many DMUs won't be comparable on all output dimensions.
- **DMU heterogeneity:** each DMU has a latent "scale" factor (lognormal) reused for both inputs and outputs, so larger/more productive DMUs use more input. This creates realistic slack patterns.
- **All-zero-output guard:** the data generator enforces that every DMU has at least one nonzero output at every date, so the analysis never fails validation.

## Performance notes

- **Data generation:** ~30 seconds for 20,000 DMUs.
- **Analysis:** O(n²) in DMU count per `(date, input)` slice. With 480 slices and 20,000 DMUs, this can take 1–3 hours depending on hardware.
- **HTML building:** ~30 seconds (building Plotly charts and writing HTML).
- **File sizes:**
  - `per_dmu_all_n20000.parquet`: ~1.0 GB (float32 after downcasting).
  - `summary_all_n20000.parquet`: ~10 MB.
  - `slack_by_dmu.html`: ~63 MB (depends on `n_dmus × n_inputs` complexity).
  - `slack_share.html`: ~0.1 MB.

## Parallel execution

If the benchmark extrapolation suggests >1 hour for the full run, use the parallel version:

```bash
python run_panel_analysis_parallel.py --n-dmus 20000
```

This reimplements the `(date, input)` slice loop using `multiprocessing.Pool`, running slices in parallel across all available CPU cores. The output format and cache paths are identical to the serial version, so you can swap them interchangeably.

To control the number of worker processes:

```bash
python run_panel_analysis_parallel.py --n-dmus 20000 --workers 4
```

The parallel version must validate DMU/date set consistency before dispatching tasks and uses the same low-level `ss.analyze()` API as the serial version, ensuring identical results.

## Implementation notes

### Sparsity guarantee

The sparsity pattern is fixed per DMU: a vectorized bitmask determines which outputs are "active" once, then this mask is reused across all 12 dates. The number of active outputs per DMU is drawn from `Poisson(λ=2)` clipped to `[1, 12]`, which has a median of exactly 3. This is verified with an assertion at generation time.

### Per-input row ordering

Each input's heatmap display independently sorts DMUs by that input's mean $|\text{slack}|$ (ascending, so smallest waste at the top). When you switch inputs via the dropdown, the row order updates to reflect the new input's slack distribution. This is correct behavior — DMUs that are efficient for one input type may be inefficient for another.

### Offline viewing

Both HTML files require an internet connection to render, since they load Plotly.js from a CDN (`include_plotlyjs="cdn"`). If you need offline functionality, modify `build_html_report.py` to use `include_plotlyjs=True` instead (this embeds the full Plotly library inline, increasing file size by ~4 MB).

## Troubleshooting

### "ValueError: Missing (dmu,date,input) data" during analysis

The synthetic generator guarantees complete data (every `(dmu, date, input)` combo present exactly once), so this shouldn't happen unless `generate_panel()` is called incorrectly. Check that both `input_df` and `output_df` are being passed to `analyze_panel()`.

### Heatmap doesn't render

Check browser console for errors. If you're offline, remember that the charts require internet access to load Plotly.js from the CDN.

### Cache not found error when building reports

Run `run_panel_analysis.py` or `run_panel_analysis_parallel.py` first to generate the parquet cache.

## References

- [slackshare README](../README.md) — FDH slack theory, API, and examples.
- [test fixtures](../tests/test_panel.py) — small hand-crafted panel examples you can step through to understand the analysis.
