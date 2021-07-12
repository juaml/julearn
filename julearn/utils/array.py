# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd

from .logging import raise_error


def ensure_2d(array):
    """Ensure array is 2-dimensional

    Parameters
    ----------
    array : array-like
        The data
    Returns
    -------
    array : np.array
        A numpy array with 2 dimensions.
    """
    if array is not None and array.ndim != 2:
        if array.ndim > 2:
            raise_error(
                'Cannot force 2 dimensions when input has > 2 dimensions')
        if isinstance(array, pd.Series):
            array = array.values
        array = array.reshape(-1, 1)
    return array


def safe_select(obj, sl):
    """Select object using on slice/indexes on the second dimensions

    Parameters
    ----------
    obj : array-like (2-dimensions)
        The data
    sl : array-like of int, slice, array-like of bool
        The index
    Returns
    -------
    obj : array-like (2-dimensions)
        The data where the columns have been selected based on sl
    """
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, sl]
    else:
        return obj[:, sl]


def safe_set(obj, sl, data):
    if isinstance(obj, pd.DataFrame):
        obj.iloc[:, sl] = data
    else:
        obj[:, sl] = data
