import pandas as pd


def ensure_2d(array):
    if array is not None and array.ndim != 2:
        if isinstance(array, pd.Series):
            array = array.values
        array = array.reshape(-1, 1)
    return array


def safe_select(obj, sl):
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, sl]
    else:
        return obj[:, sl]