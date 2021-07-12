# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np


def r2_corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]**2
