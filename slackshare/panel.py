"""Panel data wrapper for long-format input/output data across dates and input types."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core as ss


def _analyze_slice(merged: pd.DataFrame, output_types: list[str]) -> tuple[pd.DataFrame, dict]:
    """Analyze one slice, handling missing (NA) DMUs gracefully.

    eligible = DMUs with fully observed input and all output columns.
    For ineligible DMUs, returns NA for x_star, slack, efficient, shares, peer.
    Aggregates (total_input, total_slack, slack_share) computed over eligible set only.
    """
    # Identify eligible DMUs: non-NA input_value AND non-NA all output columns
    eligible_mask = merged["input_value"].notna() & merged[output_types].notna().all(axis=1)

    # Initialize result columns with NaN
    out = merged.copy()
    for col in ["x_star", "slack", "efficient", "share_of_total_input", "share_of_total_slack", "peer"]:
        out[col] = np.nan

    # If any DMU is eligible, analyze the subset and backfill results by index
    if eligible_mask.any():
        eligible_df = merged[eligible_mask]
        scored, summary = ss.analyze(eligible_df, input_col="input_value", output_cols=output_types, id_col="dmu")
        # Assign results back to the full frame using index alignment
        out.loc[eligible_df.index, "x_star"] = scored["x_star"].values
        out.loc[eligible_df.index, "slack"] = scored["slack"].values
        out.loc[eligible_df.index, "efficient"] = scored["efficient"].values
        out.loc[eligible_df.index, "share_of_total_input"] = scored["share_of_total_input"].values
        out.loc[eligible_df.index, "share_of_total_slack"] = scored["share_of_total_slack"].values
        out.loc[eligible_df.index, "peer"] = scored["peer"].values
    else:
        # No eligible DMUs: return zero sums and NA ratio
        summary = {
            "total_input": 0.0,
            "total_slack": 0.0,
            "slack_share": np.nan,
        }
        return out, summary

    return out, summary


def _check_no_duplicates(df: pd.DataFrame, type_col: str) -> None:
    """Check that no (dmu, date, type) combo appears more than once."""
    duplicates = df.set_index(["dmu", "date", type_col]).index[
        df.set_index(["dmu", "date", type_col]).index.duplicated()
    ]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate rows in {type_col} data: {duplicates.unique().tolist()}")


def analyze_panel(input_df: pd.DataFrame, output_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze FDH slack across panel: (date, input_type) slices on (date) × all output_types.

    Handles unbalanced (incomplete) panel data by gracefully handling missing inputs or outputs:
    - DMUs with missing input, missing output, or both are marked as ineligible for a slice.
    - Ineligible DMU rows receive NA values for x_star, slack, efficient, and share metrics.
    - Aggregate metrics (total_input, total_slack, slack_share) are computed over the observed
      (eligible) subset only, using skip-NA semantics: one incomplete DMU does not invalidate
      the slice's aggregates.

    Args:
        input_df: long-format inputs with columns (dmu, date, input, value).
                  Rows may be missing for some (dmu, date, input) combinations (treated as NA);
                  explicit NaN in value is also treated as missing.
        output_df: long-format outputs with columns (dmu, date, output, value).
                   Rows may be missing for some (dmu, date, output) combinations;
                   explicit NaN in value is also treated as missing.
                   Must cover the same set of dmus and dates as input_df (at the set level only;
                   individual per-type coverage may be incomplete).

    Returns:
        (per_dmu_df, summary_df):
        - per_dmu_df: concatenated results across all (date, input) slices,
          with columns: dmu, date, input, all output columns,
          x_star, slack, efficient, share_of_total_input, share_of_total_slack, peer.
          Rows for ineligible (incomplete) DMUs contain NA in the FDH result columns.
          peer is the DMU identifier that achieved x_star for this DMU; among ties, alphabetically first.
        - summary_df: one row per (date, input) slice with columns:
          date, input, total_input, total_slack, slack_share.
          Aggregates are computed over eligible DMUs only. slack_share is NA if no DMU is eligible.

    Raises:
        ValueError: if top-level data validation fails:
          - Empty input_df or output_df.
          - DMU sets do not match (both sets must have the same DMUs, though per-type coverage may differ).
          - Date sets do not match (both must cover the same dates, though per-type coverage may differ).
          - Duplicate rows (same dmu, date, type appears more than once — ambiguous which value to use).
          - Negative input values.
          - Negative output values.
          - Any DMU with all-zero outputs in a slice (violates FDH assumptions even if present).
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

    dmus = sorted(input_dmus)
    input_types = sorted(input_df["input"].unique())
    output_types = sorted(output_df["output"].unique())

    # Check for duplicates (still an error — ambiguous value)
    _check_no_duplicates(input_df, "input")
    _check_no_duplicates(output_df, "output")

    # Check for negative values (structural constraint, not a missingness issue)
    if (input_df["value"] < 0).any():
        raise ValueError("input values must not be negative")
    if (output_df["value"] < 0).any():
        raise ValueError("output columns must not contain negative values")

    # Pivot outputs by date, ensuring all output_types appear as columns (even if all-NA for some)
    output_by_date = {
        date: group.pivot(index="dmu", columns="output", values="value").reindex(index=dmus, columns=output_types)
        for date, group in output_df.groupby("date")
    }

    # Groupby input for fast lookup; use .get() to gracefully return all-NA for missing slices
    input_by_date_type_dict = {
        key: group.set_index("dmu")["value"].reindex(dmus) for key, group in input_df.groupby(["date", "input"])
    }

    per_dmu_frames = []
    summary_rows = []

    for date in sorted(output_by_date.keys()):
        out_wide = output_by_date[date]

        for input_name in input_types:
            # Use .get() to return all-NaN Series if this (date, input) slice is completely missing
            in_slice = input_by_date_type_dict.get((date, input_name), pd.Series(np.nan, index=dmus))

            merged = out_wide.copy()
            merged.insert(0, "input_value", in_slice)
            merged = merged.reset_index()

            # Check for all-zero outputs in the slice (after dropping ineligible DMUs)
            # to provide a clear error message
            eligible_mask = merged["input_value"].notna() & merged[output_types].notna().all(axis=1)
            if eligible_mask.any():
                eligible_outputs = merged.loc[eligible_mask, output_types]
                if (eligible_outputs == 0).all(axis=1).any():
                    raise ValueError("output columns must not be all zero for any DMU")

            # Analyze this slice (handles NA gracefully)
            scored, summary = _analyze_slice(merged, output_types)

            scored["date"] = date
            scored["input"] = input_name
            per_dmu_frames.append(scored)
            summary_rows.append({**summary, "date": date, "input": input_name})

    per_dmu_all = pd.concat(per_dmu_frames, ignore_index=True)
    summary_all = pd.DataFrame(summary_rows)

    return per_dmu_all, summary_all
