"""Provide tests for the scorer's registry."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest
from sklearn.metrics import make_scorer

from julearn.scoring import get_scorer, register_scorer, reset_scorer_register
from julearn.utils.typing import DataLike, EstimatorLike


def _return_1(
    estimator: EstimatorLike, X: DataLike, y: DataLike  # noqa: N803
) -> float:
    """Return 1."""
    return 1


def test_register_scorer() -> None:
    """Test registering scorers."""
    with pytest.raises(ValueError, match="useless is not a valid scorer"):
        get_scorer("useless")
    register_scorer("useless", make_scorer(_return_1))
    _ = get_scorer("useless")

    register_scorer("useless", make_scorer(_return_1), True)

    with pytest.warns(
        RuntimeWarning, match="scorer named useless already exists. "
    ):
        register_scorer("useless", make_scorer(_return_1), None)

    with pytest.raises(
        ValueError, match="scorer named useless already exists and"
    ):
        register_scorer("useless", make_scorer(_return_1), False)
    reset_scorer_register()


def test_reset_scorer() -> None:
    """Test resetting the scorers registry."""
    with pytest.raises(ValueError, match="useless is not a valid scorer "):
        get_scorer("useless")
    register_scorer("useless", make_scorer(_return_1))
    get_scorer("useless")
    reset_scorer_register()
    with pytest.raises(ValueError, match="useless is not a valid scorer "):
        get_scorer("useless")
