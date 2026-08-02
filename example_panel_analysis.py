#!/usr/bin/env python3
"""Minimal example: analyze panel data from shared_data/ and generate HTML reports."""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from slackshare.panel import analyze_panel

# Load CSVs and rename columns to match API expectations
inputs = pd.read_csv("shared_data/panel_inputs.csv")
outputs = pd.read_csv("shared_data/panel_outputs.csv")

inputs = inputs.rename(columns={
    "dmu_id": "dmu",
    "period": "date",
    "input_name": "input",
    "input_value": "value"
})

outputs = outputs.rename(columns={
    "dmu_id": "dmu",
    "period": "date",
    "output_name": "output",
    "output_value": "value"
})

# Analyze
per_dmu, summary = analyze_panel(inputs, outputs)

# Compute slack_by_dmu: aggregate slack across all (date, input) slices
slack_by_dmu = per_dmu.groupby("dmu").agg({
    "slack": "sum",
    "input_value": "sum"
}).reset_index()
slack_by_dmu.columns = ["dmu", "total_slack", "total_input"]
slack_by_dmu["slack_share"] = slack_by_dmu["total_slack"] / slack_by_dmu["total_input"]
slack_by_dmu = slack_by_dmu.sort_values("total_slack", ascending=False)

# Show results
print("Slack by DMU (top 15):")
print(slack_by_dmu.head(15))
print("\nSummary by (date, input):")
print(summary)

# ============================================================================
# GENERATE HTML REPORTS
# ============================================================================

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

input_names = sorted(per_dmu["input"].unique())
dates = sorted(per_dmu["date"].unique())
date_labels = [pd.Timestamp(d).strftime("%Y-%m") for d in dates]
dmu_names = sorted(per_dmu["dmu"].unique())

print("\n" + "="*60)
print("Generating HTML reports...")
print("="*60)

# ---- Heatmap Report ----
heatmap_traces = []

for input_name in input_names:
    input_data = per_dmu[per_dmu["input"] == input_name].copy()

    # Sort DMUs by ascending mean |slack|
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
    heatmap_data = heatmap_data.abs()
    heatmap_data = heatmap_data.reindex(index=dmu_order, columns=dates)

    # Pivot peer identities for tooltip
    peer_data = input_data.pivot_table(
        index="dmu", columns="date", values="peer", aggfunc="first"
    ).reindex(index=dmu_order, columns=dates)

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

heatmap_path = out_dir / "slack_by_dmu_shared_data.html"
fig_heatmap.write_html(str(heatmap_path), include_plotlyjs=True, full_html=True)
heatmap_size_mb = heatmap_path.stat().st_size / (1024 * 1024)
print(f"✓ Heatmap: {heatmap_path} ({heatmap_size_mb:.1f} MB)")

# ---- Bar + Line Chart Report ----
bar_traces = []
line_traces = []

for input_name in input_names:
    input_summary = summary[summary["input"] == input_name].sort_values("date")
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

bar_path = out_dir / "slack_share_shared_data.html"
fig_bar.write_html(str(bar_path), include_plotlyjs=True, full_html=True)
bar_size_mb = bar_path.stat().st_size / (1024 * 1024)
print(f"✓ Bar chart: {bar_path} ({bar_size_mb:.1f} MB)")

print("\n✓ Done! Open the HTML files in your browser:")
print(f"  - {heatmap_path}")
print(f"  - {bar_path}")
