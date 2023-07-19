"""Provides tests for the stratified groups K-fold generator."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
from numpy.testing._private.utils import assert_array_equal
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    # RepeatedStratifiedGroupKFold,  # Need sklearn #24247
)
from collections import Counter


from julearn.model_selection.continuous_stratified_kfold import (
    RepeatedContinuousStratifiedKFold,
    ContinuousStratifiedKFold,
    # RepeatedContinuousStratifiedGroupKFold,  # Need in sklearn #24247
    ContinuousStratifiedGroupKFold,
)


def test_continuous_stratified_kfold_binning() -> None:
    """Test continuous stratified K-fold generator using binning."""
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    n_bins = 5
    edges = np.histogram_bin_edges(y, bins=n_bins)
    bins = np.digitize(y, bins=edges[:-1])
    assert len(np.unique(bins)) == n_bins  # We have n_bins bins

    skcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = ContinuousStratifiedKFold(
        n_bins=n_bins, n_splits=3, shuffle=True, random_state=42
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    skcv = RepeatedStratifiedKFold(n_repeats=4, n_splits=3, random_state=42)
    jucv = RepeatedContinuousStratifiedKFold(
        n_bins=n_bins, n_repeats=4, n_splits=3, random_state=42
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)


def test_continuous_stratified_kfold_quantile() -> None:
    """Test continuous stratified K-fold generator using binning."""
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.normal(size=n_samples)
    n_bins = 5
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(y, bins=edges[:-1])

    assert len(np.unique(bins)) == n_bins  # We have n_bins bins

    # Each bin has the same number of samples
    assert all(v == n_samples / n_bins for k, v in Counter(bins).items())

    skcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = ContinuousStratifiedKFold(
        method="quantile",
        n_bins=n_bins,
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    skcv = RepeatedStratifiedKFold(n_repeats=4, n_splits=3, random_state=42)
    jucv = RepeatedContinuousStratifiedKFold(
        method="quantile",
        n_bins=n_bins,
        n_repeats=4,
        n_splits=3,
        random_state=42,
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins), jucv.split(X, y)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)


def test_continuous_stratified_group_kfold_binning() -> None:
    """Test continuous stratified group K-fold generator using binning."""
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    n_bins = 5
    edges = np.histogram_bin_edges(y, bins=n_bins)
    bins = np.digitize(y, bins=edges[:-1])
    assert len(np.unique(bins)) == n_bins  # We have n_bins bins

    groups = np.random.randint(0, 50, size=n_samples)

    skcv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = ContinuousStratifiedGroupKFold(
        n_bins=n_bins, n_splits=3, shuffle=True, random_state=42
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins, groups=groups), jucv.split(X, y, groups=groups)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    # skcv = RepeatedStratifiedKGroupFold(
    #    n_repeats=4, n_splits=3, random_state=42)
    # jucv = RepeatedSContinuousStratifiedKGroupFold(
    #     n_bins=n_bins, n_repeats=4, n_splits=3, random_state=42
    # )
    # for (sk_train, sk_test), (ju_train, ju_test) in zip(
    #     skcv.split(X, bins), jucv.split(X, y)
    # ):
    #     assert_array_equal(sk_train, ju_train)
    #     assert_array_equal(sk_test, ju_test)


def test_continuous_stratified_group_kfold_quantile() -> None:
    """Test continuous stratified group K-fold generator using binning."""
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.normal(size=n_samples)
    n_bins = 5
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(y, bins=edges[:-1])

    assert len(np.unique(bins)) == n_bins  # We have n_bins bins

    # Each bin has the same number of samples
    assert all(v == n_samples / n_bins for k, v in Counter(bins).items())

    groups = np.random.randint(0, 50, size=n_samples)

    skcv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = ContinuousStratifiedGroupKFold(
        method="quantile",
        n_bins=n_bins,
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
        skcv.split(X, bins, groups=groups), jucv.split(X, y, groups=groups)
    ):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    # skcv = RepeatedStratifiedKFold(n_repeats=4, n_splits=3, random_state=42)
    # jucv = RepeatedContinuousStratifiedKFold(
    #     method="quantile",
    #     n_bins=n_bins,
    #     n_repeats=4,
    #     n_splits=3,
    #     random_state=42,
    # )
    # for (sk_train, sk_test), (ju_train, ju_test) in zip(
    #     skcv.split(X, bins), jucv.split(X, y)
    # ):
    #     assert_array_equal(sk_train, ju_train)
    #     assert_array_equal(sk_test, ju_test)
