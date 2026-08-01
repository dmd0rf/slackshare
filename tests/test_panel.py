import pandas as pd
import numpy as np
import pytest

import slackshare as ss


@pytest.fixture
def panel_data_happy():
    """2 dates × 2 input types × 2 output types, 3 DMUs. All data present and valid."""
    input_df = pd.DataFrame({
        "dmu": ["A", "B", "C"] * 4,
        "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3 +
                [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
        "input": (["labor"] * 6) + (["capital"] * 6),
        "value": [10, 12, 8, 11, 13, 9, 5, 6, 4, 5.5, 6.5, 4.5],
    })

    output_df = pd.DataFrame({
        "dmu": ["A", "B", "C"] * 4,
        "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3 +
                [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
        "output": (["revenue"] * 6) + (["quality"] * 6),
        "value": [100, 120, 90, 110, 130, 100, 8, 9, 7, 8.5, 9.5, 7.5],
    })

    return input_df, output_df


def test_analyze_panel_happy_path(panel_data_happy):
    input_df, output_df = panel_data_happy
    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # Shape checks
    assert len(per_dmu) == 3 * 2 * 2  # 3 DMUs × 2 dates × 2 input types
    assert len(summary) == 2 * 2  # 2 dates × 2 input types
    assert "dmu" in per_dmu.columns
    assert "date" in per_dmu.columns
    assert "input" in per_dmu.columns
    assert "x_star" in per_dmu.columns
    assert "slack" in per_dmu.columns
    assert "efficient" in per_dmu.columns
    assert "share_of_total_input" in per_dmu.columns
    assert "share_of_total_slack" in per_dmu.columns
    assert "revenue" in per_dmu.columns
    assert "quality" in per_dmu.columns

    # Summary has expected columns
    assert "date" in summary.columns
    assert "input" in summary.columns
    assert "total_input" in summary.columns
    assert "total_slack" in summary.columns
    assert "slack_share" in summary.columns

    # Spot-check: no NaNs for this valid data
    assert not per_dmu["x_star"].isna().any()
    assert not summary["total_input"].isna().any()


def test_mismatched_dmus():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "C"],  # C instead of B
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [100, 110],
    })
    with pytest.raises(ValueError, match="DMU sets do not match"):
        ss.analyze_panel(input_df, output_df)


def test_mismatched_dates():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-02")] * 2,  # Different date
        "output": ["revenue"] * 2,
        "value": [100, 110],
    })
    with pytest.raises(ValueError, match="Date sets do not match"):
        ss.analyze_panel(input_df, output_df)


def test_missing_input_row():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [100, 110],
    })
    # Add another input type to input_df for only one DMU
    input_df = pd.concat([
        input_df,
        pd.DataFrame({
            "dmu": ["A"],
            "date": [pd.Timestamp("2024-01-01")],
            "input": ["capital"],
            "value": [5],
        }),
    ], ignore_index=True)
    # Missing (B, 2024-01-01, capital)
    with pytest.raises(ValueError, match="Missing"):
        ss.analyze_panel(input_df, output_df)


def test_duplicate_output_rows():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B", "A"],  # A appears twice for revenue
        "date": [pd.Timestamp("2024-01-01")] * 3,
        "output": ["revenue", "revenue", "revenue"],
        "value": [100, 110, 105],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        ss.analyze_panel(input_df, output_df)


def test_nan_value_in_input():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, np.nan],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [100, 110],
    })
    with pytest.raises(ValueError, match="NaN"):
        ss.analyze_panel(input_df, output_df)


def test_negative_input_value():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, -5],  # Negative input
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [100, 110],
    })
    with pytest.raises(ValueError, match="must not be negative"):
        ss.analyze_panel(input_df, output_df)


def test_all_zero_output_raises():
    """If any DMU has all-zero outputs in a slice, analyze_panel raises ValueError."""
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [0, 0],  # all-zero outputs
    })
    with pytest.raises(ValueError, match="must not be all zero"):
        ss.analyze_panel(input_df, output_df)


def test_empty_dataframes():
    input_df = pd.DataFrame({"dmu": [], "date": [], "input": [], "value": []})
    output_df = pd.DataFrame({"dmu": [], "date": [], "output": [], "value": []})
    with pytest.raises(ValueError, match="must not be empty"):
        ss.analyze_panel(input_df, output_df)


def test_negative_output_propagates():
    input_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "input": ["labor"] * 2,
        "value": [10, 12],
    })
    output_df = pd.DataFrame({
        "dmu": ["A", "B"],
        "date": [pd.Timestamp("2024-01-01")] * 2,
        "output": ["revenue"] * 2,
        "value": [100, -5],  # negative output
    })
    with pytest.raises(ValueError, match="must not contain negative"):
        ss.analyze_panel(input_df, output_df)
