"""Provides tests for the metrics module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pytest

from julearn.scoring.metrics import ensure_1d, r2_corr, r_corr


def test_ensure_1d() -> None:
    """Test ensure_1d."""
    y = [1, 2, 3, 4]
    assert np.all(ensure_1d(y) == y)

    y = [[1, 2, 3, 4]]
    assert np.all(ensure_1d(y) == y[0])

    with pytest.raises(ValueError, match="cannot be converted to 1d"):
        ensure_1d([[1, 2, 3, 4], [2, 3, 4, 5]])


def test_r2_corr() -> None:
    """Test r2_corr."""
    assert r2_corr([1, 2, 3, 4], [1, 2, 3, 4]) == 1
    assert r2_corr([1, 2, 3, 4], [2, 3, 4, 5]) == 1


def test_r_corr() -> None:
    """Test r_corr."""
    assert r_corr([1, 2, 3, 4], [1, 2, 3, 4]) == 1
    assert r_corr([1, 2, 3, 4], [2, 3, 4, 5]) == 1
