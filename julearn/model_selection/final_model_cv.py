"""CV Wrapper that includes a fold with all the data."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import TYPE_CHECKING, Generator, Optional, Tuple

import numpy as np


if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator


class _JulearnFinalModelCV:
    """Final model cross-validation iterator.

    Wraps any CV iterator to provide an extra iteration with the full dataset.

    Parameters
    ----------
    cv : BaseCrossValidator
        The cross-validation iterator to wrap.

    """

    def __init__(self, cv: "BaseCrossValidator") -> None:
        self.cv = cv
        if hasattr(cv, "n_repeats"):
            self.n_repeats = cv.n_repeats

    def split(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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
        This CV Splitter will generate an extra fold where the full dataset is
        used for training and testing. This is useful to train the final model
        on the full dataset at the same time as the cross-validation,
        profitting for joblib calls.

        """
        # For the first fold, train on all samples and return only 2 for test
        all_inds = np.arange(len(X))
        yield all_inds, all_inds[:2]

        yield from self.cv.split(X, y, groups)

    def get_n_splits(self) -> int:
        """Get the number of splits.

        Returns
        -------
        int
            The number of splits.

        """
        return self.cv.get_n_splits() + 1

    def __repr__(self) -> str:
        """Return the representation of the object.

        Returns
        -------
        str
            The representation of the object.

        """
        return f"{self.cv} (incl. final model)"
