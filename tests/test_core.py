import pandas as pd
import pytest

import slackshare as ss


@pytest.fixture
def sample_data():
    # DMU A: dominated by D (D has output>=10... wait check below)
    # Constructed by hand so expected results are easy to verify.
    return pd.DataFrame({
        "dmu": ["A", "B", "C", "D"],
        "emissions": [100, 80, 120, 60],
        "output": [10, 10, 15, 8],
    })


def test_fdh_scores_basic(sample_data):
    scored = ss.fdh_scores(sample_data, input_col="emissions", output_cols=["output"])

    # D: output=8. Dominators (output>=8): A(10),B(10),C(15),D(8) -> all four.
    # min emissions among these = min(100,80,120,60) = 60 -> D is self-efficient.
    d = scored.loc[scored["dmu"] == "D"].iloc[0]
    assert d["x_star"] == 60
    assert d["slack"] == 0
    assert d["efficient"]

    # A: output=10. Dominators (output>=10): A(10),B(10),C(15) -> emissions 100,80,120
    # min = 80 -> x_star=80, slack = 100-80 = 20
    a = scored.loc[scored["dmu"] == "A"].iloc[0]
    assert a["x_star"] == 80
    assert a["slack"] == 20
    assert not a["efficient"]

    # B: output=10. Same dominator set as A -> x_star=80, slack=0 (B itself achieves min)
    b = scored.loc[scored["dmu"] == "B"].iloc[0]
    assert b["x_star"] == 80
    assert b["slack"] == 0
    assert b["efficient"]

    # C: output=15. Dominators (output>=15): only C(15) -> x_star=120, slack=0
    c = scored.loc[scored["dmu"] == "C"].iloc[0]
    assert c["x_star"] == 120
    assert c["slack"] == 0
    assert c["efficient"]


def test_aggregate(sample_data):
    scored = ss.fdh_scores(sample_data, input_col="emissions", output_cols=["output"])
    summary = ss.aggregate(scored, input_col="emissions")

    assert summary["total_input"] == 100 + 80 + 120 + 60
    assert summary["total_slack"] == 20  # only A has slack
    assert summary["slack_share"] == pytest.approx(20 / 360)


def test_dmu_shares(sample_data):
    scored = ss.fdh_scores(sample_data, input_col="emissions", output_cols=["output"])
    shares = ss.dmu_shares(scored, input_col="emissions")

    a = shares.loc[shares["dmu"] == "A"].iloc[0]
    assert a["share_of_total_input"] == pytest.approx(20 / 360)
    assert a["share_of_total_slack"] == pytest.approx(1.0)  # A is the only slack

    b = shares.loc[shares["dmu"] == "B"].iloc[0]
    assert b["share_of_total_input"] == 0
    assert b["share_of_total_slack"] == 0


def test_analyze_convenience(sample_data):
    per_dmu, summary = ss.analyze(sample_data, input_col="emissions", output_cols=["output"])
    assert "share_of_total_slack" in per_dmu.columns
    assert summary["total_slack"] == 20


def test_negative_input_raises():
    bad = pd.DataFrame({"emissions": [-1, 2], "output": [1, 2]})
    with pytest.raises(ValueError):
        ss.fdh_scores(bad, input_col="emissions", output_cols=["output"])


def test_single_dmu_is_efficient():
    single = pd.DataFrame({"emissions": [50], "output": [5]})
    scored = ss.fdh_scores(single, input_col="emissions", output_cols=["output"])
    assert scored.iloc[0]["slack"] == 0
    assert scored.iloc[0]["efficient"]


# ---------------------------------------------------------------------
# Vector (multi-output) dominance
# ---------------------------------------------------------------------

@pytest.fixture
def vector_data():
    # Two outputs: revenue, quality. A unit only dominates another if it is
    # >= on BOTH dimensions.
    return pd.DataFrame({
        "dmu":      ["A", "B", "C", "D"],
        "emissions": [100, 80, 90, 60],
        "revenue":   [10, 10, 12, 8],
        "quality":   [5, 6, 4, 5],
    })


def test_vector_dominance_basic(vector_data):
    scored = ss.fdh_scores(
        vector_data, input_col="emissions", output_cols=["revenue", "quality"]
    )

    # A: (10,5). Dominators need revenue>=10 AND quality>=5:
    #   A(10,5) yes, B(10,6) yes, C(12,4) no (quality 4<5), D(8,5) no (revenue 8<10)
    #   -> emissions {100(A), 80(B)} -> x_star=80, slack=20
    a = scored.loc[scored["dmu"] == "A"].iloc[0]
    assert a["x_star"] == 80
    assert a["slack"] == 20

    # B: (10,6). Dominators need revenue>=10 AND quality>=6:
    #   A(10,5) no, B(10,6) yes, C(12,4) no, D(8,5) no
    #   -> only B itself -> x_star=80, slack=0
    b = scored.loc[scored["dmu"] == "B"].iloc[0]
    assert b["x_star"] == 80
    assert b["slack"] == 0
    assert b["efficient"]

    # C: (12,4). Dominators need revenue>=12 AND quality>=4:
    #   only C itself (12>=12, 4>=4) -> x_star=90, slack=0
    c = scored.loc[scored["dmu"] == "C"].iloc[0]
    assert c["x_star"] == 90
    assert c["slack"] == 0
    assert c["efficient"]

    # D: (8,5). Dominators need revenue>=8 AND quality>=5:
    #   A(10,5) yes, B(10,6) yes, C(12,4) no, D(8,5) yes
    #   -> emissions {100,80,60} -> x_star=60, slack=0 (D achieves the min itself)
    d = scored.loc[scored["dmu"] == "D"].iloc[0]
    assert d["x_star"] == 60
    assert d["slack"] == 0
    assert d["efficient"]


def test_vector_dominance_aggregate(vector_data):
    per_dmu, summary = ss.analyze(
        vector_data, input_col="emissions", output_cols=["revenue", "quality"]
    )
    assert summary["total_slack"] == 20  # only A has slack
    assert summary["total_input"] == 100 + 80 + 90 + 60

    a = per_dmu.loc[per_dmu["dmu"] == "A"].iloc[0]
    assert a["share_of_total_slack"] == pytest.approx(1.0)


def test_single_element_output_list(vector_data):
    scored = ss.fdh_scores(vector_data, input_col="emissions", output_cols=["revenue"])
    assert len(scored) == len(vector_data)
    assert "x_star" in scored.columns


def test_positional_args_rejected(sample_data):
    with pytest.raises(TypeError):
        ss.fdh_scores(sample_data, "emissions", "output")


def test_empty_data_raises():
    empty = pd.DataFrame({"emissions": [], "output": []})
    with pytest.raises(ValueError):
        ss.fdh_scores(empty, input_col="emissions", output_cols=["output"])


def test_all_zero_input_raises_on_aggregate():
    all_zero = pd.DataFrame({"emissions": [0, 0, 0], "output": [1, 2, 3]})
    scored = ss.fdh_scores(all_zero, input_col="emissions", output_cols=["output"])
    with pytest.raises(ZeroDivisionError):
        ss.aggregate(scored, input_col="emissions")


def test_zero_output_allowed():
    data = pd.DataFrame({
        "dmu": ["A", "B"],
        "input": [100, 80],
        "revenue": [10, 5],
        "quality": [5, 0],
    })
    scored = ss.fdh_scores(data, input_col="input", output_cols=["revenue", "quality"])
    assert len(scored) == 2
    # B has quality=0 but revenue=5, so not all-zero, should process normally


def test_negative_output_raises():
    data = pd.DataFrame({
        "dmu": ["A", "B"],
        "input": [100, 80],
        "output": [10, -5],
    })
    with pytest.raises(ValueError, match="must not contain negative"):
        ss.fdh_scores(data, input_col="input", output_cols=["output"])


def test_all_zero_output_single_column_raises():
    data = pd.DataFrame({
        "dmu": ["A", "B"],
        "input": [100, 80],
        "output": [10, 0],
    })
    with pytest.raises(ValueError, match="must not be all zero"):
        ss.fdh_scores(data, input_col="input", output_cols=["output"])


def test_all_zero_output_multiple_columns_raises():
    data = pd.DataFrame({
        "dmu": ["A", "B", "C"],
        "input": [100, 80, 90],
        "revenue": [10, 0, 5],
        "quality": [5, 0, 3],
    })
    with pytest.raises(ValueError, match="must not be all zero"):
        ss.fdh_scores(data, input_col="input", output_cols=["revenue", "quality"])


def test_all_efficient_shares_are_zero():
    data = pd.DataFrame({
        "dmu": ["A", "B", "C"],
        "input": [10, 12, 8],
        "output": [5, 6, 4],
    })
    scored = ss.fdh_scores(data, input_col="input", output_cols=["output"])
    per_dmu, summary = ss.dmu_shares(scored, input_col="input"), ss.aggregate(scored, input_col="input")

    assert summary["total_slack"] == 0
    assert (per_dmu["share_of_total_slack"] == 0.0).all()  # no NaN
