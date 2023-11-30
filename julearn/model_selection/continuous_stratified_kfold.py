"""Stratified-Groups K-Fold cross validators for regression problems."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection._split import (
    StratifiedGroupKFold,
    StratifiedKFold,
    _RepeatedSplits,
)

from ..utils import raise_error


def _discretize_y(
    method: str,
    y: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    discrete_y = None
    if method == "binning":
        bins = np.histogram_bin_edges(y, bins=n_bins)
    elif method == "quantile":
        bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    else:
        raise_error(
            f"Unknown y discreatization method {method}. ",
            ValueError,
        )
    discrete_y = np.digitize(y, bins=bins[:-1])
    return discrete_y


class ContinuousStratifiedKFold(StratifiedKFold):
    """Stratified K-Fold cross validator for regression problems.

    Stratification is done based on the discretization of the target variable
    into a fixed number of bins/quantiles.

    Parameters
    ----------
    n_bins : int
        Number of bins/quantiles to use.
    method : str, default="binning"
        Method used to stratify the groups. Can be either "binning" or
        "quantile". In the first case, the groups are stratified by binning
        the target variable. In the second case, the groups are stratified
        by quantiling the target variable.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """

    def __init__(
        self,
        n_bins,
        method="binning",
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):
        self.n_bins = n_bins
        if method not in ["binning", "quantile"]:
            raise_error(
                "The method parameter must be either 'binning' or 'quantile'.",
            )
        self.method = method
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
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

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        discrete_y = _discretize_y(self.method, y, self.n_bins)
        return super().split(X, discrete_y, groups)


class RepeatedContinuousStratifiedKFold(_RepeatedSplits):
    """Repeated Contionous Stratified K-Fold cross validator.

    Repeats :class:`julearn.model_selection.ContinuousStratifiedKFold`
    n times with different randomization in each repetition.

    Parameters
    ----------
    n_bins : int
        Number of bins/quantiles to use.
    method : str, default="binning"
        Method used to stratify the groups. Can be either "binning" or
        "quantile". In the first case, the groups are stratified by binning
        the target variable. In the second case, the groups are stratified
        by quantiling the target variable.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an intege
    """

    def __init__(
        self,
        n_bins,
        method="binning",
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            ContinuousStratifiedKFold,
            n_bins=n_bins,
            method=method,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )


class ContinuousStratifiedGroupKFold(StratifiedGroupKFold):
    """Stratified Group K-Fold cross validator for regression problems.

    Stratified Group K-Fold, where stratification is done based on the
    discretization of the target variable into a fixed number of
    bins/quantiles.

    Parameters
    ----------
    n_bins : int
        Number of bins/quantiles to use.
    method : str, default="binning"
        Method used to stratify the groups. Can be either "binning" or
        "quantile". In the first case, the groups are stratified by binning
        the target variable. In the second case, the groups are stratified
        by quantiling the target variable.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """

    def __init__(
        self,
        n_bins,
        method="binning",
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):
        self.n_bins = n_bins
        if method not in ["binning", "quantile"]:
            raise_error(
                "The method parameter must be either 'binning' or 'quantile'.",
            )
        self.method = method
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
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

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        discrete_y = _discretize_y(self.method, y, self.n_bins)
        return super().split(X, discrete_y, groups)


class RepeatedContinuousStratifiedGroupKFold(_RepeatedSplits):
    """Repeated Stratified-Groups K-Fold cross validator.

    Repeats :class:`julearn.model_selection.ContinuousStratifiedGroupKFold`
    n times with different randomization in each repetition.

    Parameters
    ----------
    n_bins : int
        Number of bins/quantiles to use.
    method : str, default="binning"
        Method used to stratify the groups. Can be either "binning" or
        "quantile". In the first case, the groups are stratified by binning
        the target variable. In the second case, the groups are stratified
        by quantiling the target variable.
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
        n_bins,
        method="binning",
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            ContinuousStratifiedGroupKFold,
            n_bins=n_bins,
            method=method,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
