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
    cv_mdsum = "maradona"
    scores1 = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "repeat": np.arange(10) // 5,
            "score": data1,
        }
    )
    scores1["cv_mdsum"] = cv_mdsum
    scores1["n_train"] = 100
    scores1["n_test"] = 20
    scores2 = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "repeat": np.arange(10) // 5,
            "score": data2,
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
        assert "p-val-corrected" not in out
        assert "model_1" in out
        assert "model_2" in out
        assert "model_0" in out["model_1"].values
        assert "model_1" in out["model_2"].values
