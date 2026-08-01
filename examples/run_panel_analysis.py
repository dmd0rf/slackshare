"""Run FDH analysis on generated panel data; cache results to parquet."""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import slackshare as ss
from generate_panel import generate_panel

DEFAULT_OUT_DIR = Path(__file__).parent / "output"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "cache"


def run_analysis(n_dmus=20000, seed=0, out_dir=DEFAULT_OUT_DIR, force=False):
    """
    Generate panel -> analyze_panel -> cache to parquet.
    Returns (per_dmu_all, summary_all) DataFrames and elapsed time.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_dmu_path = cache_dir / f"per_dmu_all_n{n_dmus}.parquet"
    summary_path = cache_dir / f"summary_all_n{n_dmus}.parquet"

    # Check cache
    if per_dmu_path.exists() and summary_path.exists() and not force:
        print(f"Loading cached results from {cache_dir}")
        per_dmu_all = pd.read_parquet(per_dmu_path)
        summary_all = pd.read_parquet(summary_path)
        return per_dmu_all, summary_all, 0.0

    print(f"Generating panel: n_dmus={n_dmus}")
    input_df, output_df = generate_panel(n_dmus=n_dmus, seed=seed)
    print(f"  Input rows: {len(input_df):,}")
    print(f"  Output rows: {len(output_df):,}")

    print("Running analyze_panel (this is O(n²) and may take time)...")
    t0 = time.perf_counter()
    per_dmu_all, summary_all = ss.analyze_panel(input_df, output_df)
    elapsed = time.perf_counter() - t0

    print(f"✓ Analysis complete in {elapsed:.1f}s")
    print(f"  Per-DMU rows: {len(per_dmu_all):,}")
    print(f"  Summary rows: {len(summary_all):,}")

    # Downcast to float32 for storage efficiency
    float_cols = ["slack", "x_star", "share_of_total_input", "share_of_total_slack"]
    for col in float_cols:
        if col in per_dmu_all.columns:
            per_dmu_all[col] = per_dmu_all[col].astype(np.float32)

    # Also downcast output columns (they were float64 from the pivot)
    output_cols = [c for c in per_dmu_all.columns if c.startswith("output_")]
    for col in output_cols:
        per_dmu_all[col] = per_dmu_all[col].astype(np.float32)

    # Summary float columns
    summary_float_cols = ["total_input", "total_slack", "slack_share"]
    for col in summary_float_cols:
        if col in summary_all.columns:
            summary_all[col] = summary_all[col].astype(np.float32)

    # Write to cache
    print(f"Writing cache to {cache_dir}")
    per_dmu_all.to_parquet(per_dmu_path, index=False)
    summary_all.to_parquet(summary_path, index=False)

    return per_dmu_all, summary_all, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FDH panel analysis and cache results")
    parser.add_argument("--n-dmus", type=int, default=20000, help="Number of DMUs (default 20000)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    parser.add_argument("--benchmark", action="store_true", help="Quick run with 500 DMUs (ignores --n-dmus)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    n_dmus = 500 if args.benchmark else args.n_dmus

    if args.benchmark:
        print("=" * 70)
        print("BENCHMARK MODE (500 DMUs)")
        print("=" * 70)
        print("Expected runtime scales as O(n²), so time(20000) ≈ time(500) × 1600.")
        print()

    per_dmu_all, summary_all, elapsed = run_analysis(
        n_dmus=n_dmus,
        seed=args.seed,
        out_dir=args.out_dir,
        force=args.force,
    )

    if elapsed > 0:
        if args.benchmark:
            scaled_time = elapsed * 1600
            print()
            print(f"Benchmark time: {elapsed:.1f}s")
            print(f"Estimated time for n_dmus=20000: ~{scaled_time/60:.1f} minutes ({scaled_time:.0f}s)")
            if scaled_time > 3600:
                print("⚠ Warning: full run may exceed 1 hour. Consider multiprocessing fallback.")
