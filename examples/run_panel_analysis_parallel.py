"""Optional parallel version of analyze_panel using multiprocessing for 480 slices."""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import slackshare as ss
from generate_panel import generate_panel

DEFAULT_OUT_DIR = Path(__file__).parent / "output"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "cache"


def _analyze_one_slice(args):
    """Analyze one (date, input) slice. Called in a worker process."""
    date, input_name, out_wide, in_slice, output_types, dmus = args

    merged = out_wide.copy()
    merged.insert(0, "input_value", in_slice)
    merged = merged.reset_index()

    scored, summary = ss.analyze(merged, input_col="input_value", output_cols=output_types)

    scored["date"] = date
    scored["input"] = input_name
    summary["date"] = date
    summary["input"] = input_name

    return scored, summary


def analyze_panel_parallel(input_df, output_df, num_workers=None):
    """
    Parallel version of analyze_panel using multiprocessing.
    Returns (per_dmu_all, summary_all) in the same shape as the serial version.
    """
    if num_workers is None:
        num_workers = None  # Use default (number of CPUs)

    input_dmus = set(input_df["dmu"].unique())
    output_dmus = set(output_df["dmu"].unique())
    if input_dmus != output_dmus:
        raise ValueError(f"DMU sets do not match: input {input_dmus}, output {output_dmus}")

    input_dates = set(input_df["date"].unique())
    output_dates = set(output_df["date"].unique())
    if input_dates != output_dates:
        raise ValueError(f"Date sets do not match: input {input_dates}, output {output_dates}")

    dmus = sorted(input_dmus)
    dates = sorted(input_dates)
    input_types = sorted(input_df["input"].unique())
    output_types = sorted(output_df["output"].unique())

    print(f"Preparing parallel tasks: {len(dates)} dates × {len(input_types)} inputs = {len(dates) * len(input_types)} slices")

    # Pivot outputs by date once
    output_by_date = {
        date: group.pivot(index="dmu", columns="output", values="value").reindex(sorted(dmus))
        for date, group in output_df.groupby("date")
    }

    # Build input by (date, input)
    input_by_date_type = {
        key: group.set_index("dmu")["value"].reindex(sorted(dmus))
        for key, group in input_df.groupby(["date", "input"])
    }

    # Prepare task args
    tasks = []
    for date in sorted(output_by_date.keys()):
        out_wide = output_by_date[date]
        for input_name in input_types:
            in_slice = input_by_date_type[(date, input_name)]
            tasks.append((date, input_name, out_wide, in_slice, output_types, dmus))

    # Run in parallel
    print(f"Running {len(tasks)} slices in parallel ({num_workers} workers)...")
    with Pool(num_workers) as pool:
        results = pool.map(_analyze_one_slice, tasks)

    # Reassemble
    per_dmu_frames = [scored for scored, _ in results]
    summary_rows = [summary for _, summary in results]

    per_dmu_all = pd.concat(per_dmu_frames, ignore_index=True)
    summary_all = pd.DataFrame(summary_rows)

    return per_dmu_all, summary_all


def run_analysis_parallel(n_dmus=20000, seed=0, out_dir=DEFAULT_OUT_DIR, force=False, num_workers=None):
    """
    Generate panel -> analyze_panel (parallel) -> cache to parquet.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_dmu_path = cache_dir / f"per_dmu_all_n{n_dmus}.parquet"
    summary_path = cache_dir / f"summary_all_n{n_dmus}.parquet"

    if per_dmu_path.exists() and summary_path.exists() and not force:
        print(f"Loading cached results from {cache_dir}")
        per_dmu_all = pd.read_parquet(per_dmu_path)
        summary_all = pd.read_parquet(summary_path)
        return per_dmu_all, summary_all, 0.0

    print(f"Generating panel: n_dmus={n_dmus}")
    input_df, output_df = generate_panel(n_dmus=n_dmus, seed=seed)
    print(f"  Input rows: {len(input_df):,}")
    print(f"  Output rows: {len(output_df):,}")

    print("Running analyze_panel in parallel...")
    t0 = time.perf_counter()
    per_dmu_all, summary_all = analyze_panel_parallel(input_df, output_df, num_workers=num_workers)
    elapsed = time.perf_counter() - t0

    print(f"✓ Analysis complete in {elapsed:.1f}s")
    print(f"  Per-DMU rows: {len(per_dmu_all):,}")
    print(f"  Summary rows: {len(summary_all):,}")

    # Downcast to float32
    float_cols = ["slack", "x_star", "share_of_total_input", "share_of_total_slack"]
    for col in float_cols:
        if col in per_dmu_all.columns:
            per_dmu_all[col] = per_dmu_all[col].astype(np.float32)

    output_cols = [c for c in per_dmu_all.columns if c.startswith("output_")]
    for col in output_cols:
        per_dmu_all[col] = per_dmu_all[col].astype(np.float32)

    summary_float_cols = ["total_input", "total_slack", "slack_share"]
    for col in summary_float_cols:
        if col in summary_all.columns:
            summary_all[col] = summary_all[col].astype(np.float32)

    print(f"Writing cache to {cache_dir}")
    per_dmu_all.to_parquet(per_dmu_path, index=False)
    summary_all.to_parquet(summary_path, index=False)

    return per_dmu_all, summary_all, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FDH panel analysis in parallel")
    parser.add_argument("--n-dmus", type=int, default=20000, help="Number of DMUs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    per_dmu_all, summary_all, elapsed = run_analysis_parallel(
        n_dmus=args.n_dmus,
        seed=args.seed,
        out_dir=args.out_dir,
        force=args.force,
        num_workers=args.workers,
    )
