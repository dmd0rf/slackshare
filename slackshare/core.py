"""FDH slack analysis for a single undesirable input.

x_i*    = min{ x_j : y_j >= y_i on every output dim }   (FDH-efficient input)
slack_i = x_i - x_i*                                    (avoidable excess)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fdh_scores(data: pd.DataFrame, *, input_col: str, output_cols: list[str]) -> pd.DataFrame:
    """Per-DMU x_star, slack, efficient. Raises if input_col has negative values."""
    if len(data) == 0:
        raise ValueError("data must contain at least one DMU (row)")

    x = data[input_col].to_numpy(dtype=float)
    y = data[output_cols].to_numpy(dtype=float)

    if np.isnan(x).any():
        raise ValueError(f"'{input_col}' contains NaN values")
    if np.isnan(y).any():
        raise ValueError("output columns contain NaN values")
    if (x < 0).any():
        raise ValueError(f"'{input_col}' contains negative values")
    if (y < 0).any():
        raise ValueError("output columns must not contain negative values")
    if (y == 0).all(axis=1).any():
        raise ValueError("output columns must not be all zero for any DMU")

    x_star = np.array([x[np.all(y >= y[i], axis=1)].min() for i in range(len(x))])

    out = data.copy()
    out["x_star"] = x_star
    out["slack"] = x - x_star
    out["efficient"] = out["slack"] == 0
    return out


def aggregate(scored: pd.DataFrame, *, input_col: str) -> dict:
    """total_input, total_slack, slack_share across all DMUs."""
    total_input = float(scored[input_col].sum())
    total_slack = float(scored["slack"].sum())
    return {
        "total_input": total_input,
        "total_slack": total_slack,
        "slack_share": total_slack / total_input,
    }


def dmu_shares(scored: pd.DataFrame, *, input_col: str) -> pd.DataFrame:
    """Adds share_of_total_input and share_of_total_slack per DMU."""
    agg = aggregate(scored, input_col=input_col)
    out = scored.copy()
    out["share_of_total_input"] = out["slack"] / agg["total_input"]
    if agg["total_slack"] == 0:
        out["share_of_total_slack"] = 0.0
    else:
        out["share_of_total_slack"] = out["slack"] / agg["total_slack"]
    return out


def analyze(data: pd.DataFrame, *, input_col: str, output_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """fdh_scores + dmu_shares + aggregate in one call."""
    scored = fdh_scores(data, input_col=input_col, output_cols=output_cols)
    return dmu_shares(scored, input_col=input_col), aggregate(scored, input_col=input_col)
