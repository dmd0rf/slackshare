import numpy as np
import pandas as pd
import pytest

import slackshare as ss


@pytest.fixture
def panel_data_happy():
    """2 dates × 2 input types × 2 output types, 3 DMUs. All data present and valid."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "C"] * 4,
            "date": [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-01-02")] * 3
            + [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-01-02")] * 3,
            "input": (["labor"] * 6) + (["capital"] * 6),
            "value": [10, 12, 8, 11, 13, 9, 5, 6, 4, 5.5, 6.5, 4.5],
        }
    )

    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "C"] * 4,
            "date": [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-01-02")] * 3
            + [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-01-02")] * 3,
            "output": (["revenue"] * 6) + (["quality"] * 6),
            "value": [100, 120, 90, 110, 130, 100, 8, 9, 7, 8.5, 9.5, 7.5],
        }
    )

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
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "C"],  # C instead of B
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    with pytest.raises(ValueError, match="DMU sets do not match"):
        ss.analyze_panel(input_df, output_df)


def test_mismatched_dates():
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-02")] * 2,  # Different date
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    with pytest.raises(ValueError, match="Date sets do not match"):
        ss.analyze_panel(input_df, output_df)


def test_missing_input_row():
    """Missing input for a (dmu, date, input_type) produces NA for that DMU in that slice."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "A"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "input": ["labor", "labor", "capital"],
            "value": [10, 12, 5],
        }
    )
    # B missing capital input
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # Slice (2024-01-01, labor): both A and B eligible (have both labor input and revenue output)
    labor_slice = per_dmu[per_dmu["input"] == "labor"]
    assert not labor_slice.loc[labor_slice["dmu"] == "A", "x_star"].isna().any()
    assert not labor_slice.loc[labor_slice["dmu"] == "B", "x_star"].isna().any()

    # Slice (2024-01-01, capital): only A eligible; B's row has NA x_star
    capital_slice = per_dmu[per_dmu["input"] == "capital"]
    assert not capital_slice.loc[capital_slice["dmu"] == "A", "x_star"].isna().any()
    assert capital_slice.loc[capital_slice["dmu"] == "B", "x_star"].isna().all()


def test_duplicate_output_rows():
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "A"],  # A appears twice for revenue
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "output": ["revenue", "revenue", "revenue"],
            "value": [100, 110, 105],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        ss.analyze_panel(input_df, output_df)


def test_nan_value_in_input():
    """Explicit NaN in input value is treated as missing; row receives NA results."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, np.nan],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # A is eligible (input 10, output 100), B is not (input NaN)
    a_row = per_dmu[per_dmu["dmu"] == "A"].iloc[0]
    b_row = per_dmu[per_dmu["dmu"] == "B"].iloc[0]

    assert not pd.isna(a_row["x_star"])
    assert pd.isna(b_row["x_star"])
    assert pd.isna(b_row["slack"])
    assert pd.isna(b_row["efficient"])


def test_negative_input_value():
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, -5],  # Negative input
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    with pytest.raises(ValueError, match="must not be negative"):
        ss.analyze_panel(input_df, output_df)


def test_all_zero_output_raises():
    """If any DMU has all-zero outputs in a slice, analyze_panel raises ValueError."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [0, 0],  # all-zero outputs
        }
    )
    with pytest.raises(ValueError, match="must not be all zero"):
        ss.analyze_panel(input_df, output_df)


def test_empty_dataframes():
    input_df = pd.DataFrame({"dmu": [], "date": [], "input": [], "value": []})
    output_df = pd.DataFrame({"dmu": [], "date": [], "output": [], "value": []})
    with pytest.raises(ValueError, match="must not be empty"):
        ss.analyze_panel(input_df, output_df)


def test_negative_output_propagates():
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, -5],  # negative output
        }
    )
    with pytest.raises(ValueError, match="must not contain negative"):
        ss.analyze_panel(input_df, output_df)


def test_missing_one_output_type():
    """Missing one output type for a DMU at a date makes that DMU ineligible across all input types at that date."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"] * 2,
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "input": ["labor", "labor", "capital", "capital"],
            "value": [10, 12, 5, 6],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "A"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "output": ["revenue", "revenue", "quality"],
            "value": [100, 110, 8],
        }
    )
    # B is missing quality output at this date

    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # Both slices should have B with NA results
    for input_type in ["labor", "capital"]:
        slice_df = per_dmu[per_dmu["input"] == input_type]
        a_row = slice_df[slice_df["dmu"] == "A"].iloc[0]
        b_row = slice_df[slice_df["dmu"] == "B"].iloc[0]
        assert not pd.isna(a_row["x_star"]), f"A should be eligible in {input_type} slice"
        assert pd.isna(b_row["x_star"]), f"B should be ineligible in {input_type} slice (missing output)"


def test_fully_empty_slice():
    """Slice with no eligible DMUs (all missing) produces NAs and zero aggregates."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [np.nan, np.nan],  # both missing
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # All per-DMU results should be NA
    assert per_dmu["x_star"].isna().all()
    assert per_dmu["slack"].isna().all()

    # Aggregates: total_input/slack are 0, slack_share is NA
    summary_row = summary.iloc[0]
    assert summary_row["total_input"] == 0.0
    assert summary_row["total_slack"] == 0.0
    assert pd.isna(summary_row["slack_share"])


def test_aggregate_skip_na_correctness():
    """Aggregate metrics skip missing DMUs: one incomplete DMU among several doesn't invalidate the slice."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "C"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "input": ["labor"] * 3,
            "value": [100, np.nan, 50],  # B is missing
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "C"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "output": ["revenue"] * 3,
            "value": [10, 10, 10],  # all have same output
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # B should have NA results
    b_row = per_dmu[per_dmu["dmu"] == "B"].iloc[0]
    assert pd.isna(b_row["x_star"])

    # A and C eligible; both have output=10, dominators={A,C} → x* = min(100,50) = 50
    a_row = per_dmu[per_dmu["dmu"] == "A"].iloc[0]
    c_row = per_dmu[per_dmu["dmu"] == "C"].iloc[0]
    assert a_row["x_star"] == 50
    assert a_row["slack"] == 50  # 100 - 50
    assert c_row["x_star"] == 50
    assert c_row["slack"] == 0  # 50 - 50

    # Aggregates over A and C only: total_input=150, total_slack=50, slack_share=50/150
    summary_row = summary.iloc[0]
    assert summary_row["total_input"] == pytest.approx(150.0)
    assert summary_row["total_slack"] == pytest.approx(50.0)
    assert summary_row["slack_share"] == pytest.approx(50.0 / 150.0)


def test_missing_entire_output_type():
    """Output type with zero rows for every DMU at a date → all DMUs ineligible, no KeyError."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [10, 12],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [100, 110],
        }
    )
    # quality output has no rows for any DMU

    per_dmu, summary = ss.analyze_panel(input_df, output_df)

    # All DMUs should be NA for the revenue slice (have it), but column reindex ensures no KeyError
    revenue_slice = per_dmu[per_dmu["input"] == "revenue"]
    assert not revenue_slice["x_star"].isna().any()  # Both A, B eligible for revenue


def test_panel_peer_column_exists():
    """Panel results include peer column."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"] * 2,
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "input": ["labor", "labor", "capital", "capital"],
            "value": [100, 80, 50, 60],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"] * 2,
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "output": ["revenue", "revenue", "quality", "quality"],
            "value": [10, 10, 5, 6],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)
    assert "peer" in per_dmu.columns


def test_panel_peer_eligible_dmu():
    """Eligible DMUs have non-NA peer values."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "input": ["labor"] * 2,
            "value": [100, 80],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [10, 10],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)
    a_row = per_dmu[per_dmu["dmu"] == "A"].iloc[0]
    b_row = per_dmu[per_dmu["dmu"] == "B"].iloc[0]
    # Both have output=10, so both are in each other's dominance set
    # A: x_star=min(100,80)=80, peer should be "B"
    # B: x_star=min(100,80)=80, peer should be "B" (self)
    assert a_row["peer"] == "B"
    assert b_row["peer"] == "B"


def test_panel_peer_ineligible_dmu_na():
    """Ineligible DMUs have NA peer."""
    input_df = pd.DataFrame(
        {
            "dmu": ["A", "B", "A"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "input": ["labor", "labor", "capital"],
            "value": [100, np.nan, 50],
        }
    )
    output_df = pd.DataFrame(
        {
            "dmu": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "output": ["revenue"] * 2,
            "value": [10, 10],
        }
    )
    per_dmu, summary = ss.analyze_panel(input_df, output_df)
    # B should be ineligible in the labor slice (input is NA)
    labor_slice = per_dmu[per_dmu["input"] == "labor"]
    b_labor = labor_slice[labor_slice["dmu"] == "B"].iloc[0]
    assert pd.isna(b_labor["peer"])
