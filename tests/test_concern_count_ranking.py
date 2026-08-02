import pandas as pd
import pytest
import slackshare as ss


@pytest.fixture(scope="module")
def slack_and_ground_truth():
    """Load panel data, run analyze_panel, compute total slack per DMU, and load ground truth."""
    inputs = pd.read_csv("shared_data/panel_inputs.csv").rename(columns={
        "dmu_id": "dmu",
        "period": "date",
        "input_name": "input",
        "input_value": "value"
    })
    outputs = pd.read_csv("shared_data/panel_outputs.csv").rename(columns={
        "dmu_id": "dmu",
        "period": "date",
        "output_name": "output",
        "output_value": "value"
    })

    per_dmu, _ = ss.analyze_panel(inputs, outputs)

    # Aggregate slack across all inputs and dates per DMU (raw values)
    slack_by_dmu = per_dmu.groupby("dmu")["slack"].sum()

    # Load ground truth
    gt_df = pd.read_csv("shared_data/ground_truth_ranking.csv").set_index("entity_id")
    perf_rank = gt_df["performance_rank"]  # Low rank = less slack (better); high rank = more slack (worse)
    ext_metric = gt_df["external_metric"]

    return slack_by_dmu, perf_rank, ext_metric


def test_slack_ranking_vs_ground_truth_all_entities(slack_and_ground_truth):
    """Spearman correlation: raw slack vs both ground truth columns (all entities)."""
    slack_by_dmu, perf_rank, ext_metric = slack_and_ground_truth

    common = slack_by_dmu.index.intersection(perf_rank.index)

    spearman_perf = slack_by_dmu.loc[common].corr(perf_rank.loc[common], method="spearman")
    spearman_ext = slack_by_dmu.loc[common].corr(ext_metric.loc[common], method="spearman")

    print(f"\n=== ALL ENTITIES ({len(common)} DMUs overlap) ===")
    print(f"Slack vs Performance rank: {spearman_perf:.4f}")
    print(f"Slack vs External metric: {spearman_ext:.4f}")

    assert spearman_perf == pytest.approx(0.274, abs=0.01)
    assert spearman_ext == pytest.approx(0.205, abs=0.01)


def test_slack_ranking_vs_ground_truth_top50(slack_and_ground_truth):
    """Top-50 overlap: worst by slack vs worst by ground truth."""
    slack_by_dmu, perf_rank, ext_metric = slack_and_ground_truth

    # Top 50 worst by slack (largest slack values)
    top50_worst_slack = set(slack_by_dmu.nlargest(50).index)

    # Top 50 worst by performance_rank (largest rank numbers = worst)
    top50_worst_perf = set(perf_rank.nlargest(50).index)

    # Top 50 worst by external_metric (largest values)
    top50_worst_ext = set(ext_metric.nlargest(50).index)

    # Compute overlaps
    overlap_perf = len(top50_worst_slack & top50_worst_perf)
    overlap_ext = len(top50_worst_slack & top50_worst_ext)

    # Expected overlap by random chance: (50 * 50) / N_dmus
    n_dmus = len(slack_by_dmu.index.intersection(perf_rank.index))
    expected_overlap = (50 * 50) / n_dmus

    print(f"\n=== TOP 50 WORST ENTITIES ===")
    print(f"Total DMUs in common: {n_dmus}")
    print(f"Expected overlap by random chance: {expected_overlap:.1f}")
    print(f"\nOverlap (slack vs performance_rank): {overlap_perf}/50 (vs {expected_overlap:.1f} expected)")
    print(f"Overlap (slack vs external_metric): {overlap_ext}/50 (vs {expected_overlap:.1f} expected)")

    assert overlap_perf == 25, f"Expected 25 overlap with performance_rank, got {overlap_perf}"
    assert overlap_ext == 27, f"Expected 27 overlap with external_metric, got {overlap_ext}"
