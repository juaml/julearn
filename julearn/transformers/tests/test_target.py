# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from numpy.testing import assert_array_equal

from julearn.transformers.target import (TargetTransfromerWrapper,
                                         is_targettransformer)
from julearn.transformers.confounds import TargetConfoundRemover


def test_get_target_transformer():
    X = np.arange(9).reshape(3, 3)
    y = np.arange(3)

    zscore_target = TargetTransfromerWrapper(StandardScaler())
    np.random.seed(42)
    y_trans_wrapped = zscore_target.fit(X=X, y=y).transform(X=X, y=y)
    np.random.seed(42)
    y_trans_manual = (StandardScaler()
                      .fit(X=y.reshape(-1, 1))
                      .transform(X=y.reshape(-1, 1))
                      ).reshape(-1)

    assert_array_equal(y_trans_manual, y_trans_wrapped)


def test_is_targettransformer():

    assert not is_targettransformer(StandardScaler())
    assert is_targettransformer(TargetTransfromerWrapper(StandardScaler()))
    assert is_targettransformer(TargetConfoundRemover())

    with pytest.raises(ValueError,
                       match='is_targettransformer can only be applied to '):
        assert is_targettransformer(LinearRegression())
