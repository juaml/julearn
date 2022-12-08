"""Definition of scoring metrics."""
import numpy as np


def ensure_1d(y):
    """Ensure array is 1D.

    Parameters
    ----------
    y : array

    Returns
    -------
    y : array

    """
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            y = np.squeeze(y)
        if y.ndim > 1:
            raise ValueError('y must be 1d')
    return y


def r2_corr(y_true, y_pred):
    """Calculate the squared pearson correlation.

    Not to be confused with the coefficient of determination R2.

    Parameters
    ----------
    y_true : 1d numpy array
        true target values
    y_pred : 1d numpy array
        predicted target values

    Returns
    -------
    r2_corr : float
        squared pearson correlation coefficient

    """
    return np.corrcoef(ensure_1d(y_true), ensure_1d(y_pred))[0, 1]**2


def r_pearson(y_true, y_pred):
    """Calculate the pearson correlation.

    Parameters
    ----------
    y_true : 1d numpy array
        true target values
    y_pred : 1d numpy array
        predicted target values

    Returns
    -------
    r_pearson : float
        pearson correlation coefficient

    """
    return np.corrcoef(ensure_1d(y_true), ensure_1d(y_pred))[0, 1]
