"""Generate synthetic panel data: 20,000 DMUs, 40 inputs, 12 outputs (sparse, median 3)."""

import argparse
import numpy as np
import pandas as pd

N_DATES = 12
DATE_FREQ = "MS"
START_DATE = "2024-01-01"
N_INPUTS = 40
N_OUTPUTS = 12
DEFAULT_N_DMUS = 20000


def generate_panel(n_dmus=DEFAULT_N_DMUS, seed=0):
    """
    Generate synthetic long-format panel data.

    Returns (input_df, output_df) with columns (dmu, date, input/output, value).
    Every (dmu, date, input) and (dmu, date, output) combo present exactly once.
    Outputs are sparse: median 3 active outputs per DMU, structurally fixed across dates.
    All-zero-output rows are structurally impossible.
    """
    rng = np.random.RandomState(seed)

    # DMU heterogeneity: one scale factor per DMU, reused for inputs and outputs
    dmu_scale = rng.lognormal(mean=0, sigma=0.8, size=n_dmus)

    # Per-type magnitude bases
    input_base = rng.uniform(10, 1000, size=N_INPUTS)
    output_base = rng.uniform(10, 1000, size=N_OUTPUTS)

    # Time variation: seasonal + trend
    month_indices = np.arange(N_DATES)
    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * month_indices / 12)
    trend_factor = 1 + 0.02 * month_indices
    time_factor = seasonal_factor * trend_factor  # shape (12,)

    # Sparsity: n_active = clip(1 + Poisson(2), 1, 12) per DMU
    n_active = np.clip(1 + rng.poisson(2, size=n_dmus), 1, N_OUTPUTS)
    assert np.median(n_active) == 3, f"Median n_active should be 3, got {np.median(n_active)}"

    # Which outputs are active per DMU: vectorized top-k selection via rank
    # Ensure this is fixed once and reused across all dates (structural sparsity)
    perm = np.argsort(rng.random((n_dmus, N_OUTPUTS)), axis=1)
    rank = np.argsort(perm, axis=1)
    active_mask = rank < n_active[:, None]  # shape (n_dmus, N_OUTPUTS)

    # Build inputs (dense) and outputs (sparse with active_mask)
    dates = pd.date_range(start=START_DATE, periods=N_DATES, freq=DATE_FREQ)

    input_rows = []
    output_rows = []

    for d_idx, dmu in enumerate(range(n_dmus)):
        dmu_id = f"dmu_{d_idx:05d}"
        scale = dmu_scale[d_idx]

        for t_idx, date in enumerate(dates):
            time_mult = time_factor[t_idx]

            # Inputs: all 40 present for every DMU/date
            for k in range(N_INPUTS):
                noise = rng.lognormal(mean=0, sigma=0.25)
                value = scale * input_base[k] * time_mult * noise
                value = max(value, 1e-6)  # floor to avoid underflow
                input_rows.append({
                    "dmu": dmu_id,
                    "date": date,
                    "input": f"input_{k:02d}",
                    "value": value,
                })

            # Outputs: only where active_mask[d_idx, o] is True
            for o in range(N_OUTPUTS):
                if active_mask[d_idx, o]:
                    noise = rng.lognormal(mean=0, sigma=0.3)
                    value = scale * output_base[o] * time_mult * noise
                    value = max(value, 1e-6)
                else:
                    value = 0.0
                output_rows.append({
                    "dmu": dmu_id,
                    "date": date,
                    "output": f"output_{o:02d}",
                    "value": value,
                })

    # Convert to DataFrames with categorical dtypes (saves ~80% memory vs object strings)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic FDH panel data")
    parser.add_argument("--n-dmus", type=int, default=DEFAULT_N_DMUS, help=f"Number of DMUs (default {DEFAULT_N_DMUS})")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    args = parser.parse_args()

    print(f"Generating panel data: n_dmus={args.n_dmus}, n_inputs={N_INPUTS}, n_outputs={N_OUTPUTS}, n_dates={N_DATES}")
    input_df, output_df = generate_panel(n_dmus=args.n_dmus, seed=args.seed)

    print(f"Input rows: {len(input_df):,} (expected {args.n_dmus * N_DATES * N_INPUTS:,})")
    print(f"Output rows: {len(output_df):,} (expected {args.n_dmus * N_DATES * N_OUTPUTS:,})")

    n_active = []
    for dmu in input_df["dmu"].cat.categories:
        n_nonzero_outputs = (output_df[output_df["dmu"] == dmu]["value"] > 0).sum() // N_DATES
        n_active.append(n_nonzero_outputs)

    print(f"Sparsity check: n_active distribution = {np.histogram(n_active, bins=range(1, 14))[0]}")
    print(f"Median n_active: {np.median(n_active)}")
    print("✓ Generator OK")
