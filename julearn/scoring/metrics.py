import numpy as np


def r2_corr(y_true, y_pred):
    return np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]**2
