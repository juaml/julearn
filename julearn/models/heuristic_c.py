# Classes for heuristic C calculation for linearSVC and logistic regression.
#
# Authors: Kaustub R. Patil <k.patil@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
#
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from julearn.utils import raise_error

from ..utils.logging import logger
from ..utils.typing import ArrayLike, DataLike


def heuristic_C(data: np.ndarray) -> float:
    """Calculate the heuristic C for linearSVR (Joachims 2002).

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    C : float
        The heuristic C value.

    """

    if data is None:
        logger.error("No data was provided.")

    C = 1 / np.mean(np.sqrt((data**2).sum(axis=1)))

    # Formular Kaustubh: C = 1/mean(sqrt(rowSums(data^2)))

    return C


class LinearSVCHeuristicC(LinearSVC):
    """LinearSVC with heuristically calculated C value."""

    def __init__(self, **kwargs):
        if "C" in kwargs:
            raise_error(
                "C value provided in constructor, this is not allowed since C"
                "is calculated heuristically from the data. Please remove C"
                "from the constructor arguments."
            )
        super().__init__(**kwargs)

    # Overwrite fit method to use heuristic C as HP
    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "LinearSVCHeuristicC":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            correspond to stronger regularization for the samples. If not
            provided, all samples are given unit weight.

        Returns
        -------
        LinearSVCHeuristicC
            The fitted model.

        """
        # calculate heuristic C
        C = heuristic_C(X)
        logger.info(f"Using heuristic C = {C} for LinearSVC")

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self  # convention in scikitlearn


class LogisticRegressionHeuristicC(LogisticRegression):
    """LogisticRegression with heuristically calculated C value."""

    def __init__(self, **kwargs):
        if "C" in kwargs:
            raise_error(
                "C value provided in constructor, this is not allowed since C"
                "is calculated heuristically from the data. Please remove C"
                "from the constructor arguments."
            )
        super().__init__(**kwargs)

    # Overwrite fit method to use heuristic C as HP
    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "LogisticRegressionHeuristicC":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            correspond to stronger regularization for the samples. If not
            provided, all samples are given unit weight.

        Returns
        -------
        LogisticRegressionHeuristicC
            The fitted model.

        """
        C = heuristic_C(X)
        logger.info(f"Using heuristic C = {C} for LogisticRegression")

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
        return self
