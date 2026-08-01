"""Build separate interactive HTML reports (heatmap and bar chart) from cached analysis results."""

import argparse
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

DEFAULT_OUT_DIR = Path(__file__).parent / "output"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "cache"


def build_heatmap_report(per_dmu_all, input_names, n_dmus, out_dir):
    """Build heatmap HTML with dropdown for input selection."""
    print("Building heatmap report...")

    dmu_names = sorted(per_dmu_all["dmu"].unique())
    dates = sorted(per_dmu_all["date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%Y-%m") for d in dates]

    # ========== BUILD HEATMAP TRACES ==========
    heatmap_traces = []

    for input_name in input_names:
        # Filter to this input
        input_data = per_dmu_all[per_dmu_all["input"] == input_name].copy()

        # Sort DMUs by ascending mean |slack| (top to bottom: smallest to largest)
        dmu_mean_slack = (
            input_data.groupby("dmu")["slack"]
            .apply(lambda x: x.abs().mean())
            .sort_values(ascending=True)
        )
        dmu_order = dmu_mean_slack.index.tolist()

        # Pivot to heatmap: rows=DMU (sorted), cols=date, values=|slack|
        heatmap_data = input_data.pivot_table(
            index="dmu", columns="date", values="slack", aggfunc="first"
        )
        heatmap_data = heatmap_data.abs()  # Take absolute value
        heatmap_data = heatmap_data.reindex(index=dmu_order, columns=dates)

        # Create heatmap trace (no colorbar)
        trace = go.Heatmap(
            z=heatmap_data.values,
            x=date_labels,
            y=dmu_order,
            colorscale="Viridis",
            showscale=False,  # Hide colorbar/legend
            visible=(input_names[0] == input_name),  # First input visible by default
            name=input_name,
            hovertemplate="DMU: %{y}<br>Date: %{x}<br>|Slack|: %{z:.4f}<extra></extra>",
        )
        heatmap_traces.append(trace)

    # ========== CREATE FIGURE ==========
    fig_heatmap = go.Figure(data=heatmap_traces)

    # ========== SETUP DROPDOWN BUTTONS ==========
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

    # ========== LAYOUT ==========
    heatmap_height = max(800, 200 + len(dmu_names) * 5)  # 5px per row + margins

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

    # ========== WRITE HTML ==========
    heatmap_path = out_dir / "slack_by_dmu.html"
    fig_heatmap.write_html(str(heatmap_path), include_plotlyjs="cdn", full_html=True)
    heatmap_size_mb = heatmap_path.stat().st_size / (1024 * 1024)
    print(f"✓ Heatmap report: {heatmap_path} ({heatmap_size_mb:.1f} MB)")


def build_bar_chart_report(summary_all, input_names, out_dir):
    """Build bar chart HTML with dropdown for input selection."""
    print("Building bar chart report...")

    # ========== BUILD BAR CHART TRACES ==========
    bar_traces = []
    line_traces = []

    for input_name in input_names:
        # Filter summary to this input
        input_summary = summary_all[summary_all["input"] == input_name].sort_values("date")
        summary_dates = [pd.Timestamp(d).strftime("%Y-%m") for d in input_summary["date"]]

        # Bar trace: total input (primary y-axis)
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

        # Line trace: slackshare (secondary y-axis overlay, 0-100 range)
        line_trace = go.Scatter(
            x=summary_dates,
            y=input_summary["slack_share"] * 100,  # Convert to percentage
            name="Slackshare",
            yaxis="y2",
            mode="lines+markers",
            visible=(input_names[0] == input_name),
            line=dict(color="rgba(255, 127, 14, 1)", width=2),
            marker=dict(size=8),
            hovertemplate="Date: %{x}<br>Slackshare: %{y:.1f}%<extra></extra>",
        )
        line_traces.append(line_trace)

    # Combine traces
    all_traces = bar_traces + line_traces

    # ========== CREATE FIGURE ==========
    fig_bar = go.Figure(data=all_traces)

    # ========== SETUP DROPDOWN BUTTONS ==========
    buttons = []
    for i, input_name in enumerate(input_names):
        visible = [False] * len(all_traces)
        # Bar trace i visible
        visible[i] = True
        # Line trace (len(bar_traces) + i) visible
        visible[len(bar_traces) + i] = True
        button = dict(
            label=input_name,
            method="update",
            args=[{"visible": visible}, {"title": f"Slack Share & Total Input: {input_name}"}],
        )
        buttons.append(button)

    # ========== LAYOUT ==========
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
        yaxis=dict(
            title="Total Input",
        ),
        yaxis2=dict(
            title="Slackshare (%)",
            range=[0, 100],  # Fixed range
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

    # ========== WRITE HTML ==========
    bar_path = out_dir / "slack_share.html"
    fig_bar.write_html(str(bar_path), include_plotlyjs="cdn", full_html=True)
    bar_size_mb = bar_path.stat().st_size / (1024 * 1024)
    print(f"✓ Bar chart report: {bar_path} ({bar_size_mb:.1f} MB)")


def build_reports(n_dmus=20000, out_dir=DEFAULT_OUT_DIR):
    """Build both heatmap and bar chart reports from cached analysis."""
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_dmu_path = cache_dir / f"per_dmu_all_n{n_dmus}.parquet"
    summary_path = cache_dir / f"summary_all_n{n_dmus}.parquet"

    if not per_dmu_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Cache not found. Run run_panel_analysis.py first.\n"
            f"Expected: {per_dmu_path} and {summary_path}"
        )

    print(f"Reading cache from {cache_dir}")
    per_dmu_all = pd.read_parquet(per_dmu_path)
    summary_all = pd.read_parquet(summary_path)

    print(f"Per-DMU rows: {len(per_dmu_all):,}")
    print(f"Summary rows: {len(summary_all):,}")

    # Get metadata
    input_names = sorted(per_dmu_all["input"].unique())
    dmu_names = sorted(per_dmu_all["dmu"].unique())

    print(f"DMUs: {len(dmu_names)}, Inputs: {len(input_names)}")

    # Build both reports
    build_heatmap_report(per_dmu_all, input_names, n_dmus, out_dir)
    build_bar_chart_report(summary_all, input_names, out_dir)

    print(f"\n✓ Both reports generated in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build separate interactive HTML reports")
    parser.add_argument("--n-dmus", type=int, default=20000, help="Number of DMUs (must match cache)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    args = parser.parse_args()

    build_reports(n_dmus=args.n_dmus, out_dir=args.out_dir)
