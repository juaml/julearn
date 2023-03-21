"""Provides tests for the stratified groups K-fold generator."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from julearn.model_selection.stratified_groups_kfold import (
    StratifiedGroupsKFold,
    RepeatedStratifiedGroupsKFold,
)
import numpy as np
from numpy.testing._private.utils import assert_array_equal
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


def test_stratified_groups_kfold() -> None:
    """Test stratified groups K-fold generator."""
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    bins = np.digitize(y, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])

    skcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = StratifiedGroupsKFold(n_splits=3, shuffle=True, random_state=42)
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y, groups=bins)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    skcv = RepeatedStratifiedKFold(n_repeats=4, n_splits=3, random_state=42)
    jucv = RepeatedStratifiedGroupsKFold(
        n_repeats=4, n_splits=3, random_state=42
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y, groups=bins)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)
