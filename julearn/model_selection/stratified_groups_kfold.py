"""Stratified-Groups K-Fold cross validators for regression problems."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection._split import StratifiedKFold, _RepeatedSplits
from sklearn.utils.validation import check_array

from ..utils import raise_error


class StratifiedGroupsKFold(StratifiedKFold):
    """Stratified-Groups K-Fold cross validator for regression problems.

    Stratified Groups K-Fold n times with different randomization in
    each repetition. This particular implementation ensures that the
    groups are stratified. That is, each group is represented in each
    fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : object
            The stratification variable.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If the `groups` parameter is None.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        if groups is None:
            raise_error("The groups parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        return super().split(X, groups, None)


class RepeatedStratifiedGroupsKFold(_RepeatedSplits):
    """Repeated Stratified-Groups K-Fold cross validator.

    Repeats :class:`julearn.model_selection.StratifiedGroupsKFold` n times with
    different randomization in each repetition.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an intege
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            StratifiedGroupsKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
