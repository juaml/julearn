import numpy as np


def ensure_1d(y):
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            y = np.squeeze(y)
        if y.ndim > 1:
            raise ValueError('y must be 1d')
    return y


def r2_corr(y_true, y_pred):
    return np.corrcoef(ensure_1d(y_true), ensure_1d(y_pred))[0, 1]**2
