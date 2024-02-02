"""Metrics for evaluating models."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
from numpy.typing import ArrayLike


def ensure_1d(y: ArrayLike) -> np.ndarray:
    """Ensure that y is 1d.

    Parameters
    ----------
    y : ArrayLike
        The array to be checked.

    Returns
    -------
    np.ndarray
        The array as a 1d numpy array.

    Raises
    ------
    ValueError
        If y cannot be converted to a 1d numpy array.

    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if y.ndim > 1:
        y = np.squeeze(y)
    if y.ndim > 1:
        raise ValueError("y cannot be converted to 1d")
    return y


def r2_corr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute squared Pearson product-moment correlation coefficient.

    Parameters
    ----------
    y_true : ArrayLike
        The true values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        The squared Pearson product-moment correlation coefficient.

    """
    return np.corrcoef(ensure_1d(y_true), ensure_1d(y_pred))[0, 1] ** 2


def r_corr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Pearson product-moment correlation coefficient.

    Parameters
    ----------
    y_true : ArrayLike
        The true values.
    y_pred : ArrayLike
        The predicted values.

    Returns
    -------
    float
        Pearson product-moment correlation coefficient.

    """

    return np.corrcoef(ensure_1d(y_true), ensure_1d(y_pred))[0, 1]
