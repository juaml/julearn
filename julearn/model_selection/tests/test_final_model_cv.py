"""Provides tests for the final model CV."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import RepeatedStratifiedKFold

from julearn.model_selection.final_model_cv import _JulearnFinalModelCV
from julearn.utils import _compute_cvmdsum


def test_final_model_cv() -> None:
    """Test the final model CV."""
    sklearn_cv = RepeatedStratifiedKFold(
        n_repeats=2, n_splits=5, random_state=42
    )

    julearn_cv = _JulearnFinalModelCV(sklearn_cv)

    assert julearn_cv.get_n_splits() == 11
    assert julearn_cv.n_repeats == 2

    n_features = 10
    n_samples = 123
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    all_ju = list(julearn_cv.split(X, y))
    all_sk = list(sklearn_cv.split(X, y))

    assert len(all_ju) == len(all_sk) + 1
    for i in range(1, 11):
        assert_array_equal(all_ju[i][0], all_sk[i-1][0])
        assert_array_equal(all_ju[i][1], all_sk[i-1][1])

    assert all_ju[0][0].shape[0] == n_samples
    assert all_ju[0][1].shape[0] == 2
    assert_array_equal(all_ju[0][0], np.arange(n_samples))


def test_final_model_cv_mdsum() -> None:
    """Test the mdsum of the final model CV."""
    sklearn_cv = RepeatedStratifiedKFold(
        n_repeats=2, n_splits=5, random_state=42
    )

    julearn_cv = _JulearnFinalModelCV(sklearn_cv)

    mdsum = _compute_cvmdsum(julearn_cv)
    mdsum_sk = _compute_cvmdsum(sklearn_cv)
    assert mdsum == mdsum_sk
