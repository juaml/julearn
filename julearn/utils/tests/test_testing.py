import numpy as np
from numpy.testing import assert_array_equal
from julearn.utils.testing import PassThroughTransformer
from julearn.utils.testing import TargetPassThroughTransformer


def test_passthrough():

    X = np.arange(9).reshape(3, 3)
    y = np.arange(3)

    X_t = PassThroughTransformer().fit_transform(X, y)
    y_t = TargetPassThroughTransformer().fit_transform(X, y)

    assert_array_equal(X, X_t)
    assert_array_equal(y, y_t)
