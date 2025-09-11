"""Class for Stratified Bootstrap cross-validation iterator."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

from typing import ClassVar, Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection._split import (
    _build_repr,
    _UnsupportedGroupCVMixin,
)
from sklearn.utils import check_array, metadata_routing
from sklearn.utils._array_api import _convert_to_numpy, get_namespace
from sklearn.utils.extmath import _approximate_mode
from sklearn.utils.metadata_routing import _MetadataRequester
from sklearn.utils.validation import (
    _num_samples,
    check_random_state,
    indexable,
)


class StratifiedBootstrap(_MetadataRequester, _UnsupportedGroupCVMixin):
    """Class-wise stratified bootstrap cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class in `y` in a
    binary or multiclass classification setting.

    Parameters
    ----------
    n_splits : int, default=200
        Number of bootstrap iterations.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split: ClassVar = {"groups": metadata_routing.UNUSED}

    def __init__(
        self,
        n_splits: int = 200,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X,  # noqa: N803
        y,
        groups=None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
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
        # From BaseCrossValidator
        X, y, groups = indexable(X, y, groups)
        # From StratifiedShuffleSplit
        n_samples = _num_samples(X)
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)

        # Convert to numpy as not all operations are supported by the Array
        # API. `y` is probably never a very large array, which means that
        # converting it should be cheap
        xp, _ = get_namespace(y)
        y = _convert_to_numpy(y, xp=xp)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class in y has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2."
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"),
            np.cumsum(class_counts)[:-1],
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_samples, rng)

            train = []
            test = []

            # Adapted from mlxtend.BootstrapOutOfBag
            for i in range(n_classes):
                train_idx = rng.choice(
                    class_indices[i], size=n_i[i], replace=True
                )
                test_idx = np.array(
                    list(set(class_indices[i]) - set(train_idx))
                )
                train.extend(train_idx)
                test.extend(test_idx)

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def get_n_splits(
        self,
        X=None,  # noqa: N803
        y=None,
        groups=None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        return self.n_splits  # pragma: no cover

    def __repr__(self) -> str:
        """Object representation."""
        return _build_repr(self)  # pragma: no cover
