"""
Minimal end-to-end FDH slack analysis example.

Generates a synthetic panel (500 DMUs, 40 inputs, 12 outputs, 12 dates),
runs FDH analysis via slackshare, and builds two interactive Plotly HTML reports.

Run:
    python run_example.py              # default 500 DMUs
    python run_example.py --n-dmus 100 # smaller, faster
    python run_example.py --n-dmus 2000 # larger, slower (O(n²))
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import slackshare as ss

# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_panel(n_dmus=500, seed=0):
    """Generate synthetic long-format panel data: 500 DMUs, 40 inputs, 12 outputs (sparse), 12 dates."""
    rng = np.random.RandomState(seed)

    N_DATES = 12
    DATE_FREQ = "MS"
    START_DATE = "2024-01-01"
    N_INPUTS = 40
    N_OUTPUTS = 12

    # DMU heterogeneity: one scale factor per DMU
    dmu_scale = rng.lognormal(mean=0, sigma=0.8, size=n_dmus)

    # Per-type magnitude bases
    input_base = rng.uniform(10, 1000, size=N_INPUTS)
    output_base = rng.uniform(10, 1000, size=N_OUTPUTS)

    # Time variation: seasonal + trend
    month_indices = np.arange(N_DATES)
    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * month_indices / 12)
    trend_factor = 1 + 0.02 * month_indices
    time_factor = seasonal_factor * trend_factor

    # Sparsity: n_active = clip(1 + Poisson(2), 1, 12) per DMU
    n_active = np.clip(1 + rng.poisson(2, size=n_dmus), 1, N_OUTPUTS)

    # Which outputs are active per DMU: fixed per DMU across all dates
    perm = np.argsort(rng.random((n_dmus, N_OUTPUTS)), axis=1)
    rank = np.argsort(perm, axis=1)
    active_mask = rank < n_active[:, None]

    dates = pd.date_range(start=START_DATE, periods=N_DATES, freq=DATE_FREQ)

    input_rows = []
    output_rows = []

    for d_idx, _dmu in enumerate(range(n_dmus)):
        dmu_id = f"dmu_{d_idx:05d}"
        scale = dmu_scale[d_idx]

        for t_idx, date in enumerate(dates):
            time_mult = time_factor[t_idx]

            # Inputs: all 40 present for every DMU/date
            for k in range(N_INPUTS):
                noise = rng.lognormal(mean=0, sigma=0.25)
                value = scale * input_base[k] * time_mult * noise
                value = max(value, 1e-6)
                input_rows.append(
                    {
                        "dmu": dmu_id,
                        "date": date,
                        "input": f"input_{k:02d}",
                        "value": value,
                    }
                )

            # Outputs: only where active_mask[d_idx, o] is True
            for o in range(N_OUTPUTS):
                if active_mask[d_idx, o]:
                    noise = rng.lognormal(mean=0, sigma=0.3)
                    value = scale * output_base[o] * time_mult * noise
                    value = max(value, 1e-6)
                else:
                    value = 0.0
                output_rows.append(
                    {
                        "dmu": dmu_id,
                        "date": date,
                        "output": f"output_{o:02d}",
                        "value": value,
                    }
                )

    input_df = pd.DataFrame(input_rows)
    input_df["dmu"] = input_df["dmu"].astype("category")
    input_df["input"] = input_df["input"].astype("category")
    input_df["date"] = input_df["date"].astype("category")

    output_df = pd.DataFrame(output_rows)
    output_df["dmu"] = output_df["dmu"].astype("category")
    output_df["output"] = output_df["output"].astype("category")
    output_df["date"] = output_df["date"].astype("category")

    # Defensive check: no all-zero outputs per (dmu, date) slice
    per_dmu_date = output_df.groupby(["dmu", "date"])["value"].sum()
    assert (per_dmu_date > 0).all(), "Found a (dmu, date) slice with all-zero outputs"

    return input_df, output_df


# ============================================================================
# ANALYSIS
# ============================================================================


def run_analysis(input_df, output_df):
    """Run FDH slack analysis on the panel."""
    print("Running FDH analysis...")
    per_dmu_all, summary_all = ss.analyze_panel(input_df, output_df)
    return per_dmu_all, summary_all


# ============================================================================
# REPORTING: HEATMAP
# ============================================================================


def build_heatmap_report(per_dmu_all, input_names, dmu_names, out_dir):
    """Build heatmap HTML with dropdown for input selection."""
    print("Building heatmap report...")

    dates = sorted(per_dmu_all["date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%Y-%m") for d in dates]

    heatmap_traces = []

    for input_name in input_names:
        # Filter to this input
        input_data = per_dmu_all[per_dmu_all["input"] == input_name].copy()

        # Sort DMUs by ascending mean |slack|
        dmu_mean_slack = input_data.groupby("dmu")["slack"].apply(lambda x: x.abs().mean()).sort_values(ascending=True)
        dmu_order = dmu_mean_slack.index.tolist()

        # Pivot to heatmap: rows=DMU (sorted), cols=date, values=|slack|
        heatmap_data = input_data.pivot_table(index="dmu", columns="date", values="slack", aggfunc="first")
        heatmap_data = heatmap_data.abs()
        heatmap_data = heatmap_data.reindex(index=dmu_order, columns=dates)

        # Pivot peer identities for tooltip
        peer_data = input_data.pivot_table(index="dmu", columns="date", values="peer", aggfunc="first").reindex(
            index=dmu_order, columns=dates
        )

        trace = go.Heatmap(
            z=heatmap_data.values,
            x=date_labels,
            y=dmu_order,
            customdata=peer_data.values,
            colorscale="Viridis",
            showscale=False,
            visible=(input_names[0] == input_name),
            name=input_name,
            hovertemplate="DMU: %{y}<br>Date: %{x}<br>|Slack|: %{z:.4f}<br>Peer: %{customdata}<extra></extra>",
        )
        heatmap_traces.append(trace)

    # Create figure and dropdown
    fig_heatmap = go.Figure(data=heatmap_traces)

    buttons = []
    for i, input_name in enumerate(input_names):
        visible = [False] * len(input_names)
        visible[i] = True
        button = dict(
            label=input_name,
            method="update",
            args=[{"visible": visible}, {"title": f"Slack Heatmap: {input_name}"}],
        )
        buttons.append(button)

    heatmap_height = max(800, 200 + len(dmu_names) * 5)

    fig_heatmap.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="lightgray",
                bordercolor="gray",
                font=dict(size=12),
            )
        ],
        title=f"Slack by DMU: {input_names[0]}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="DMU (sorted by median |slack|, ascending)"),
        template="plotly_white",
        height=heatmap_height,
        hovermode="closest",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=100, r=50, t=120, b=60),
    )

    heatmap_path = out_dir / "slack_by_dmu.html"
    fig_heatmap.write_html(str(heatmap_path), include_plotlyjs=True, full_html=True)
    heatmap_size_mb = heatmap_path.stat().st_size / (1024 * 1024)
    print(f"✓ Heatmap: {heatmap_path} ({heatmap_size_mb:.1f} MB)")


# ============================================================================
# REPORTING: BAR + LINE CHART
# ============================================================================


def build_bar_chart_report(summary_all, input_names, out_dir):
    """Build bar + line chart HTML with dropdown for input selection."""
    print("Building bar chart report...")

    bar_traces = []
    line_traces = []

    for input_name in input_names:
        input_summary = summary_all[summary_all["input"] == input_name].sort_values("date")
        summary_dates = [pd.Timestamp(d).strftime("%Y-%m") for d in input_summary["date"]]

        bar_trace = go.Bar(
            x=summary_dates,
            y=input_summary["total_input"],
            name="Total Input",
            yaxis="y1",
            visible=(input_names[0] == input_name),
            marker=dict(color="rgba(31, 119, 180, 0.7)"),
            hovertemplate="Date: %{x}<br>Total Input: %{y:.2f}<extra></extra>",
        )
        bar_traces.append(bar_trace)

        line_trace = go.Scatter(
            x=summary_dates,
            y=input_summary["slack_share"] * 100,
            name="Slackshare",
            yaxis="y2",
            mode="lines+markers",
            visible=(input_names[0] == input_name),
            line=dict(color="rgba(255, 127, 14, 1)", width=2),
            marker=dict(size=8),
            hovertemplate="Date: %{x}<br>Slackshare: %{y:.1f}%<extra></extra>",
        )
        line_traces.append(line_trace)

    all_traces = bar_traces + line_traces
    fig_bar = go.Figure(data=all_traces)

    buttons = []
    for i, input_name in enumerate(input_names):
        visible = [False] * len(all_traces)
        visible[i] = True
        visible[len(bar_traces) + i] = True
        button = dict(
            label=input_name,
            method="update",
            args=[{"visible": visible}, {"title": f"Total Input & Slackshare: {input_name}"}],
        )
        buttons.append(button)

    fig_bar.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="lightgray",
                bordercolor="gray",
                font=dict(size=12),
            )
        ],
        title=f"Total Input & Slackshare: {input_names[0]}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Total Input"),
        yaxis2=dict(
            title="Slackshare (%)",
            range=[0, 100],
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        height=500,
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=80, r=80, t=120, b=60),
        legend=dict(x=0.5, y=-0.2, orientation="h"),
    )

    bar_path = out_dir / "slack_share.html"
    fig_bar.write_html(str(bar_path), include_plotlyjs=True, full_html=True)
    bar_size_mb = bar_path.stat().st_size / (1024 * 1024)
    print(f"✓ Bar chart: {bar_path} ({bar_size_mb:.1f} MB)")


# ============================================================================
# MAIN
# ============================================================================


def main(n_dmus=500, out_dir=None):
    """Generate panel → analyze → build reports."""
    if out_dir is None:
        out_dir = Path(__file__).parent / "output"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running FDH example (n_dmus={n_dmus})")
    print(f"Output: {out_dir}")
    print()

    # Generate
    print("Generating synthetic panel data...")
    t0 = time.perf_counter()
    input_df, output_df = generate_panel(n_dmus=n_dmus, seed=0)
    gen_time = time.perf_counter() - t0
    print(f"✓ Generated in {gen_time:.1f}s: {len(input_df):,} input rows, {len(output_df):,} output rows")

    # Analyze
    print()
    t0 = time.perf_counter()
    per_dmu_all, summary_all = run_analysis(input_df, output_df)
    ana_time = time.perf_counter() - t0
    print(f"✓ Analyzed in {ana_time:.1f}s: {len(per_dmu_all):,} per-DMU rows, {len(summary_all):,} summary rows")

    if ana_time > 60:
        print("  (Note: runtime scales as O(n²), so larger n_dmus will be much slower)")

    # Get metadata
    input_names = sorted(per_dmu_all["input"].unique())
    dmu_names = sorted(per_dmu_all["dmu"].unique())
    n_outputs = len([c for c in per_dmu_all.columns if c.startswith("output_")])
    print(f"  DMUs: {len(dmu_names)}, Inputs: {len(input_names)}, Outputs: {n_outputs}")

    # Build reports
    print()
    build_heatmap_report(per_dmu_all, input_names, dmu_names, out_dir)
    build_bar_chart_report(summary_all, input_names, out_dir)

    print()
    print("✓ Done! Open the HTML files in your browser:")
    print(f"  - {out_dir / 'slack_by_dmu.html'}")
    print(f"  - {out_dir / 'slack_share.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDH slack analysis example: generate, analyze, report")
    parser.add_argument("--n-dmus", type=int, default=500, help="Number of DMUs (default 500)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: output)")
    args = parser.parse_args()

    main(n_dmus=args.n_dmus, out_dir=args.out_dir)
