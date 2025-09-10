"""Provides tests for the stratified bootstrap CV generator."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import pytest
from sklearn.datasets import make_classification

from julearn.model_selection import StratifiedBootstrap


def test_stratified_bootstrap_error() -> None:
    """Test stratified bootstrap error."""
    X, y = make_classification(n_samples=1, n_classes=1)
    with pytest.raises(
        ValueError, match="The least populated class in y has only 1"
    ):
        cv = StratifiedBootstrap()
        list(cv.split(X, y))


@pytest.mark.parametrize(
    "n_classes, n_splits",
    [
        (2, 50),
        (2, 100),
        (2, 200),
        (3, 50),
        (3, 100),
        (3, 200),
        (4, 50),
        (4, 100),
        (4, 200),
        (5, 50),
        (5, 100),
        (5, 200),
        (6, 50),
        (6, 100),
        (6, 200),
        (7, 50),
        (7, 100),
        (7, 200),
        (8, 50),
        (8, 100),
        (8, 200),
        (9, 50),
        (9, 100),
        (9, 200),
        (10, 50),
        (10, 100),
        (10, 200),
    ],
)
def test_stratified_bootstrap(n_classes: int, n_splits: int) -> None:
    """Test stratified bootstrap CV generator.

    Parameters
    ----------
    n_classes : int
        The parametrized number of classes (or strata).
    n_splits : int
        The parametrized number of splits.

    """
    samples = 100 * n_classes
    X, y = make_classification(
        n_samples=samples,
        n_features=n_classes * 10,
        n_informative=n_classes,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
    )
    cv = StratifiedBootstrap(n_splits=n_splits, random_state=42)
    # Check splits
    results = list(cv.split(X, y))
    assert len(results) == n_splits
    # Check train and test indices are disjoint
    for train_idxs, test_idxs in cv.split(X, y):
        assert frozenset(train_idxs).isdisjoint(test_idxs)
