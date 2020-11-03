import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

import pytest

from julearn.prepare import prepare_input_data


def _check_np_input(prepared, X, y, confounds, groups):
    df_X_conf, df_y, df_groups, _ = prepared

    new_X = X
    if confounds is not None:
        new_X = np.c_[X, confounds]
    assert_array_equal(df_X_conf.values, new_X)
    assert_array_equal(df_y.values, y)
    if groups is not None:
        assert_array_equal(df_groups.values, groups)


def test_prepare_input_data_np():
    """Test validate input data (numpy)"""

    # Test X (1d) + y
    X = np.random.rand(4)
    y = np.random.rand(4)
    prepared = prepare_input_data(
        X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)
    _check_np_input(prepared, X=X[:, None], y=y, confounds=None, groups=None)

    # Test X (2d) + y
    X = np.random.rand(4, 3)
    y = np.random.rand(4)
    prepared = prepare_input_data(
        X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)
    _check_np_input(prepared, X=X, y=y, confounds=None, groups=None)

    # Test X (1d) + y + confounds (1d)
    X = np.random.rand(4)
    y = np.random.rand(4)
    confounds = np.random.rand(4)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=None, groups=None)
    _check_np_input(prepared, X=X, y=y, confounds=confounds, groups=None)

    # Test X (2d) + y + confounds (1d)
    X = np.random.rand(4, 4)
    y = np.random.rand(4)
    confounds = np.random.rand(4)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=None, groups=None)
    _check_np_input(prepared, X=X, y=y, confounds=confounds, groups=None)

    # Test X (2d) + y + confounds (2d)
    X = np.random.rand(4, 4)
    y = np.random.rand(4)
    confounds = np.random.rand(4, 2)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=None, groups=None)
    _check_np_input(prepared, X=X, y=y, confounds=confounds, groups=None)

    # Test X (2d) + y + confounds (2d) + groups
    X = np.random.rand(4, 4)
    y = np.random.rand(4)
    confounds = np.random.rand(4, 2)
    groups = np.random.rand(4)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=None, groups=groups)
    _check_np_input(prepared, X=X, y=y, confounds=confounds, groups=groups)

    # Test X (2d) + y + confounds (2d) + groups + pos_labels
    X = np.random.rand(10, 4)
    y = np.random.randint(-3, 0, 10)
    pos_labels = -1
    confounds = np.random.rand(10, 2)
    groups = np.random.rand(10)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=pos_labels,
        groups=groups)

    labeled_y = y[y == pos_labels]
    _check_np_input(prepared, X=X, y=labeled_y, confounds=confounds,
                    groups=groups)

    # Error check

    # Wrong types
    with pytest.raises(ValueError,
                       match=r"if no dataframe is specified"):
        X = dict()
        y = np.random.rand(4)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"if no dataframe is specified"):
        X = np.random.rand(4)
        y = dict()
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"if no dataframe is specified"):
        X = np.random.rand(4)
        y = np.random.rand(4)
        prepared = prepare_input_data(
            X=X, y=y, confounds=dict(), df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"if no dataframe is specified"):
        X = np.random.rand(4)
        y = np.random.rand(4)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=dict())

    # Wrong number of dimensions
    with pytest.raises(ValueError,
                       match=r"be at most bi-dimentional"):
        X = np.random.rand(4, 3, 2)
        y = np.random.rand(4)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"must be one-dimentional"):
        X = np.random.rand(4, 3)
        y = np.random.rand(4, 2)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"be at most bi-dimentional"):
        X = np.random.rand(4, 4)
        y = np.random.rand(4)
        confounds = np.random.rand(4, 2, 3)
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=None, pos_labels=None,
            groups=None)

    with pytest.raises(ValueError,
                       match=r"must be one-dimentional"):
        X = np.random.rand(4, 4)
        y = np.random.rand(4)
        groups = np.random.rand(4, 2, 3)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None,
            groups=groups)

    # Wrong dimensions
    with pytest.raises(ValueError,
                       match=r"number of samples in X do not match"):
        X = np.random.rand(4, 2)
        y = np.random.rand(3)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"number of samples in X do not match"):
        X = np.random.rand(4, 2)
        y = np.random.rand(4)
        confounds = np.random.rand(5, 2)
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=None, pos_labels=None,
            groups=None)

        with pytest.raises(ValueError,
                           match=r"number of samples in X do not match"):
            X = np.random.rand(4, 2)
            y = np.random.rand(4)
            groups = np.random.rand(5)
            prepared = prepare_input_data(
                X=X, y=y, confounds=None, df=None, pos_labels=None,
                groups=groups)