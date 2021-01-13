import numpy as np


def r2_corr(estimator, X, y):
    y_pred = estimator.predict(X)
    return np.corrcoef(y_pred, y)**2
