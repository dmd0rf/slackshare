"""FDH slack analysis for a single undesirable input.

x_i*    = min{ x_j : y_j >= y_i on every output dim }   (FDH-efficient input)
slack_i = x_i - x_i*                                    (avoidable excess)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fdh_scores(
    data: pd.DataFrame,
    *,
    input_col: str,
    output_cols: list[str],
    id_col: str | None = None,
) -> pd.DataFrame:
    """Per-DMU x_star, slack, efficient, peer. Raises if input_col has negative values.

    peer: identifier of the DMU that achieves x_star for this DMU. Among ties, chosen via
    lexicographic order: (x_value, id_string) ascending, so alphabetically first DMU name breaks ties.
    If id_col is None, uses data.index.astype(str).
    """
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

    # Extract identifiers for peer tie-breaking
    if id_col is not None:
        ids = data[id_col].astype(str).to_numpy()
    else:
        ids = data.index.astype(str).to_numpy()

    x_star = np.empty(len(x), dtype=float)
    peer = np.empty(len(x), dtype=object)

    for i in range(len(x)):
        # Dominance mask: indices j where y[j] >= y[i] on all outputs
        dominator_mask = np.all(y >= y[i], axis=1)
        dominator_indices = np.where(dominator_mask)[0]
        dominator_x = x[dominator_indices]
        dominator_ids = ids[dominator_indices]

        # Find minimum x value
        min_x_val = dominator_x.min()
        x_star[i] = min_x_val

        # Among ties (same x_star), pick alphabetically first id
        min_mask = dominator_x == min_x_val
        min_ids = dominator_ids[min_mask]
        peer[i] = sorted(min_ids)[0]

    out = data.copy()
    out["x_star"] = x_star
    out["slack"] = x - x_star
    out["efficient"] = out["slack"] == 0
    out["peer"] = peer
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


def analyze(
    data: pd.DataFrame,
    *,
    input_col: str,
    output_cols: list[str],
    id_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """fdh_scores + dmu_shares + aggregate in one call."""
    scored = fdh_scores(data, input_col=input_col, output_cols=output_cols, id_col=id_col)
    return dmu_shares(scored, input_col=input_col), aggregate(scored, input_col=input_col)
