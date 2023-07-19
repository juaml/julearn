import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from julearn.stats.corrected_ttest import (
    _compute_corrected_ttest,
    corrected_ttest,
)


def test__compute_corrected_ttest_alternatives():
    """Test the _compute_corrected_ttest function."""
    rvs1 = stats.norm.rvs(loc=0.5, scale=0.2, size=20, random_state=42)
    rvs2 = stats.norm.rvs(loc=0.51, scale=0.2, size=20, random_state=45)
    rvs3 = stats.norm.rvs(loc=0.9, scale=0.2, size=20, random_state=50)
    _, p1 = _compute_corrected_ttest(rvs1 - rvs2, n_train=70, n_test=30)

    assert p1 > 0.7

    _, p2 = _compute_corrected_ttest(rvs1 - rvs3, n_train=70, n_test=30)

    assert p2 < 0.1

    _, p3 = _compute_corrected_ttest(
        rvs1 - rvs3, n_train=70, n_test=30, alternative="less"
    )
    assert p3 < 0.05  # rvs1 is less than rvs3

    _, p4 = _compute_corrected_ttest(
        rvs1 - rvs3, n_train=70, n_test=30, alternative="greater"
    )
    assert p4 > 0.90  # rvs1 is less than rvs3, so this should be high

    with pytest.raises(ValueError, match="Invalid alternative"):
        _compute_corrected_ttest(
            rvs1 - rvs3, n_train=70, n_test=30, alternative="not_valid"
        )


def test_corrected_ttest() -> None:
    """Test the corrected_ttest function."""

    data1 = np.random.rand(10)
    data2 = np.random.rand(10) + 0.05
    data3 = np.random.rand(10) + 0.1
    cv_mdsum = "maradona"
    scores1 = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "repeat": np.arange(10) // 5,
            "test_score": data1,
        }
    )
    scores1["cv_mdsum"] = cv_mdsum
    scores1["n_train"] = 100
    scores1["n_test"] = 20
    scores2 = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "repeat": np.arange(10) // 5,
            "test_score": data2,
        }
    )
    scores2["cv_mdsum"] = cv_mdsum
    scores2["n_train"] = 100
    scores2["n_test"] = 20

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = corrected_ttest(scores1, scores2)
        assert len(out) == 1
        assert "p-val" in out
        assert "p-val-corrected" in out
        assert out["p-val-corrected"][0] == out["p-val"][0]
        assert "model_1" in out
        assert "model_2" in out
        assert "model_1" in out["model_1"].values
        assert "model_2" in out["model_2"].values

    scores3 = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "repeat": np.arange(10) // 5,
            "test_score": data3,
        }
    )
    scores3["cv_mdsum"] = cv_mdsum
    scores3["n_train"] = 100
    scores3["n_test"] = 20

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = corrected_ttest(scores1, scores2, scores3)
        assert len(out) == 3
        assert "p-val" in out
        assert "p-val-corrected" in out
        assert "model_1" in out
        assert "model_2" in out
        assert "model_1" in out["model_1"].values
        assert "model_2" in out["model_1"].values
        assert "model_2" in out["model_2"].values
        assert "model_3" in out["model_2"].values


def test_corrected_ttest_errors() -> None:
    """Test the corrected_ttest function."""

    data1 = np.random.rand(10)
    data2 = np.random.rand(10) + 0.05
    scores1 = pd.DataFrame(
        {
            "test_score": data1,
        }
    )
    scores2 = pd.DataFrame(
        {
            "test_score": data2,
        }
    )

    with pytest.raises(ValueError, match="cv_mdsum"):
        corrected_ttest(scores1, scores2)

    scores1["cv_mdsum"] = "maradona"
    scores2["cv_mdsum"] = "messi"

    with pytest.raises(ValueError, match="fold"):
        corrected_ttest(scores1, scores2)

    scores1["fold"] = np.arange(10) % 5
    scores2["fold"] = np.arange(10) % 5

    with pytest.raises(ValueError, match="repeat"):
        corrected_ttest(scores1, scores2)

    scores1["repeat"] = np.arange(10) // 5
    scores2["repeat"] = np.arange(10) // 5

    with pytest.raises(ValueError, match="n_train"):
        corrected_ttest(scores1, scores2)

    scores1["n_train"] = 100
    scores2["n_train"] = 100

    with pytest.raises(ValueError, match="n_test"):
        corrected_ttest(scores1, scores2)

    scores1["n_test"] = 90
    scores2["n_test"] = 90

    with pytest.raises(ValueError, match="different CVs"):
        corrected_ttest(scores1, scores2)

    scores1["cv_mdsum"] = "non-reproducible"
    scores2["cv_mdsum"] = "non-reproducible"

    with pytest.raises(ValueError, match="non-reproducible"):
        corrected_ttest(scores1, scores2)

    scores1["cv_mdsum"] = "maradona"
    scores2["cv_mdsum"] = "maradona"
    scores3 = scores2

    with pytest.raises(ValueError, match="two-sided"):
        corrected_ttest(scores1, scores2, scores3, alternative="wrong")

    scores1["n_train"] = [100] * 9 + [90]
    scores1["n_test"] = 90
    with pytest.warns(RuntimeWarning, match="training set"):
        corrected_ttest(scores1, scores2)

    scores1["n_train"] = 100
    scores1["n_test"] = [100] * 9 + [90]
    with pytest.warns(RuntimeWarning, match="testing set"):
        corrected_ttest(scores1, scores2)
