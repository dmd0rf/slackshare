"""Panel data wrapper for long-format input/output data across dates and input types."""

from __future__ import annotations

import pandas as pd

from . import core as ss


def _validate_panel(df: pd.DataFrame, type_col: str, dmus: set, dates: set) -> None:
    """Validate that every (dmu, date, type) combo is present exactly once with non-null value."""
    expected = pd.MultiIndex.from_product(
        [sorted(dmus), sorted(dates), sorted(df[type_col].unique())],
        names=["dmu", "date", type_col],
    )
    actual = df.set_index(["dmu", "date", type_col]).index

    missing = expected.difference(actual)
    if len(missing) > 0:
        raise ValueError(f"Missing {type_col} data: {missing.tolist()}")

    duplicates = actual[actual.duplicated()]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate rows in {type_col} data: {duplicates.unique().tolist()}")

    if df["value"].isna().any():
        raise ValueError(f"{type_col} data contains NaN values")


def analyze_panel(input_df: pd.DataFrame, output_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze FDH slack across panel: (date, input_type) slices on (date) × all output_types.

    Args:
        input_df: long-format inputs with columns (dmu, date, input, value).
                  Must have one row per unique (dmu, date, input) combination.
        output_df: long-format outputs with columns (dmu, date, output, value).
                   Must have one row per unique (dmu, date, output) combination.
                   Must cover the same set of dmus and dates as input_df.

    Returns:
        (per_dmu_df, summary_df):
        - per_dmu_df: concatenated results across all (date, input) slices,
          with columns: dmu, date, input, all output columns,
          x_star, slack, efficient, share_of_total_input, share_of_total_slack.
        - summary_df: one row per (date, input) slice with columns:
          date, input, total_input, total_slack, slack_share.

    Raises:
        ValueError: if validation fails (mismatched DMU/date sets, missing or
                    duplicate rows, NaN values, all-zero inputs, or any DMU
                    with all-zero outputs within a slice).
    """
    if len(input_df) == 0 or len(output_df) == 0:
        raise ValueError("input_df and output_df must not be empty")

    input_dmus = set(input_df["dmu"].unique())
    output_dmus = set(output_df["dmu"].unique())
    if input_dmus != output_dmus:
        raise ValueError(f"DMU sets do not match: input {input_dmus}, output {output_dmus}")

    input_dates = set(input_df["date"].unique())
    output_dates = set(output_df["date"].unique())
    if input_dates != output_dates:
        raise ValueError(f"Date sets do not match: input {input_dates}, output {output_dates}")

    dmus = input_dmus
    dates = sorted(input_dates)
    input_types = sorted(input_df["input"].unique())
    output_types = sorted(output_df["output"].unique())

    _validate_panel(input_df, "input", dmus, dates)
    _validate_panel(output_df, "output", dmus, dates)

    if (input_df["value"] < 0).any():
        raise ValueError("input values must not be negative")

    output_by_date = {
        date: group.pivot(index="dmu", columns="output", values="value").reindex(sorted(dmus))
        for date, group in output_df.groupby("date")
    }
    input_by_date_type = {
        key: group.set_index("dmu")["value"].reindex(sorted(dmus))
        for key, group in input_df.groupby(["date", "input"])
    }

    per_dmu_frames = []
    summary_rows = []

    for date in sorted(output_by_date.keys()):
        out_wide = output_by_date[date]

        for input_name in input_types:
            in_slice = input_by_date_type[(date, input_name)]

            merged = out_wide.copy()
            merged.insert(0, "input_value", in_slice)
            merged = merged.reset_index()

            scored, summary = ss.analyze(
                merged, input_col="input_value", output_cols=output_types
            )

            scored["date"] = date
            scored["input"] = input_name
            per_dmu_frames.append(scored)
            summary_rows.append({**summary, "date": date, "input": input_name})

    per_dmu_all = pd.concat(per_dmu_frames, ignore_index=True)
    summary_all = pd.DataFrame(summary_rows)

    return per_dmu_all, summary_all
