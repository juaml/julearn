"""Provides tests for the stratified bootstra CV generator."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
import pytest

from julearn.model_selection import StratifiedBootstrap


@pytest.mark.parametrize(
    "n_classes, test_size",
    [
        (3, 0.2),
        (2, 0.5),
        (4, 0.8),
    ],
)
def test_stratified_bootstrap(n_classes: int, test_size: float) -> None:
    """Test stratified bootstrap CV generator.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    test_size : float
        Test size.
    """
    n_samples = 100
    X = np.random.rand(n_samples, 2)

    y = np.random.randint(0, n_classes, n_samples)

    cv = StratifiedBootstrap(n_splits=10, test_size=test_size)

    for train, test in cv.split(X, y):
        y_train = y[train]
        y_test = y[test]
        for i in range(n_classes):
            n_y = (y == i).sum()
            n_y_train = (y_train == i).sum()
            n_y_test = (y_test == i).sum()
            assert abs(n_y_train - (n_y * (1 - test_size))) < 1
            assert abs(n_y_test - (n_y * test_size)) < 1
