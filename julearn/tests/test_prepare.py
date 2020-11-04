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

    n_features = X.shape[1] if X.ndim == 2 else 1
    feature_names = [f'feature_{i}' for i in range(n_features)]
    assert all(x in df_X_conf.columns for x in feature_names)

    if confounds is not None:
        n_confounds = confounds.shape[1] if confounds.ndim == 2 else 1
        c_names = [f'confound_{i}' for i in range(n_confounds)]
        assert all(x in df_X_conf.columns for x in c_names)


def _check_df_input(prepared, X, y, confounds, groups, df):
    df_X_conf, df_y, df_groups, _ = prepared

    assert_array_equal(df[X].values, df_X_conf[X].values)
    assert_array_equal(df_y.values, df[y].values)
    if confounds is not None:
        assert_array_equal(df[confounds].values, df_X_conf[confounds].values)
    if groups is not None:
        assert_array_equal(df[groups].values, df_groups)


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

    labeled_y = (y == pos_labels).astype(np.int)
    _check_np_input(prepared, X=X, y=labeled_y, confounds=confounds,
                    groups=groups)

    # Test X (2d) + y + confounds (2d) + groups + pos_labels (several)
    X = np.random.rand(10, 4)
    y = np.random.randint(-4, 0, 10)
    pos_labels = [-2, -1]
    confounds = np.random.rand(10, 2)
    groups = np.random.rand(10)
    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=None, pos_labels=pos_labels,
        groups=groups)

    labeled_y = np.isin(y, pos_labels).astype(np.int)
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
                       match=r"be at most bi-dimensional"):
        X = np.random.rand(4, 3, 2)
        y = np.random.rand(4)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"must be one-dimensional"):
        X = np.random.rand(4, 3)
        y = np.random.rand(4, 2)
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=None, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"be at most bi-dimensional"):
        X = np.random.rand(4, 4)
        y = np.random.rand(4)
        confounds = np.random.rand(4, 2, 3)
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=None, pos_labels=None,
            groups=None)

    with pytest.raises(ValueError,
                       match=r"must be one-dimensional"):
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


def test_prepare_input_data_df():
    """Test validate input data (dataframe)"""
    data = np.random.rand(4, 10)
    columns = [f'f_{x}' for x in range(data.shape[1])]

    # Test X (2d) + y
    X = columns[:-1]
    y = columns[-1]
    df = pd.DataFrame(data=data, columns=columns)

    prepared = prepare_input_data(
        X=X, y=y, confounds=None, df=df, pos_labels=None, groups=None)
    _check_df_input(prepared, X=X, y=y, confounds=None, groups=None, df=df)

    # Test X (2d) + y + groups
    X = columns[:5]
    y = columns[7]
    groups = columns[8]

    prepared = prepare_input_data(
        X=X, y=y, confounds=None, df=df, pos_labels=None, groups=groups)
    _check_df_input(prepared, X=X, y=y, confounds=None, groups=groups, df=df)

    # Test X (2d) + groups + confounds (2d)
    X = columns[:5]
    y = columns[6]
    groups = columns[7]
    confounds = columns[8:]

    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=df, pos_labels=None, groups=groups)
    _check_df_input(prepared, X=X, y=y, confounds=confounds, groups=groups,
                    df=df)

    # Test X (1d) + y + groups + confounds (1d)
    X = columns[2]
    y = columns[6]
    groups = columns[7]
    confounds = columns[8]

    prepared = prepare_input_data(
        X=X, y=y, confounds=confounds, df=df, pos_labels=None, groups=groups)
    _check_df_input(prepared, X=X, y=y, confounds=confounds, groups=groups,
                    df=df)

    # Wrong types
    with pytest.raises(ValueError,
                       match=r"X must be a string or list of strings"):
        X = 2
        y = columns[6]
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=None)

    with pytest.raises(ValueError, match=r"y must be a string"):
        X = columns[:5]
        y = ['bad']
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=None)

    with pytest.raises(ValueError,
                       match=r"confounds must be a string or list "):
        X = columns[:5]
        y = columns[6]
        groups = columns[7]
        confounds = 2
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None, groups=None)

    with pytest.raises(ValueError, match=r"groups must be a string"):
        X = columns[:5]
        y = columns[6]
        groups = 2
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=groups)

    with pytest.raises(ValueError, match=r"df must be a pandas.DataFrame"):
        X = columns[:5]
        y = columns[6]
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=dict(), pos_labels=None, groups=None)

    # Wrong columns
    X = columns[:5] + ['wrong']
    y = columns[6]
    groups = columns[7]
    confounds = columns[8:]
    with pytest.raises(ValueError, match=r"missing: \['wrong'\]"):
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None,
            groups=groups)

    X = columns[:5]
    y = 'wrong'
    groups = columns[7]
    confounds = columns[8:]
    with pytest.raises(ValueError, match=r"wrong"):
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None,
            groups=groups)

    X = columns[:5]
    y = columns[6]
    groups = 'wrong'
    confounds = columns[8:]
    with pytest.raises(ValueError, match=r"wrong"):
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None,
            groups=groups)

    X = columns[:5]
    y = columns[6]
    groups = None
    confounds = columns[8:] + ['wrong']
    with pytest.raises(ValueError, match=r"missing: \['wrong'\]"):
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None,
            groups=groups)

    # Test overlapping X, y, groups and confounds

    # y in X
    X = columns[:5]
    y = columns[4]
    with pytest.warns(RuntimeWarning, match='y is part of X'):
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=groups)
    _check_df_input(prepared, X=X, y=y, confounds=None, groups=groups, df=df)

    # y and groups
    X = columns[:5]
    y = columns[6]
    groups = columns[6]
    with pytest.warns(RuntimeWarning,
                      match='y and groups are the same column'):
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=groups)
        _check_df_input(prepared, X=X, y=y, confounds=None, groups=groups,
                        df=df)

    # X and groups
    X = columns[:5]
    y = columns[6]
    groups = columns[3]
    with pytest.warns(RuntimeWarning, match='groups is part of X'):
        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df, pos_labels=None, groups=groups)
        _check_df_input(prepared, X=X, y=y, confounds=None, groups=groups,
                        df=df)

    # X and confounds
    X = columns[:5]
    y = columns[-1]
    groups = None
    confounds = columns[4:9]
    overlapping = columns[4]
    with pytest.warns(RuntimeWarning, match=overlapping):
        prepared = prepare_input_data(
            X=X, y=y, confounds=confounds, df=df, pos_labels=None,
            groups=groups)
        _check_df_input(prepared, X=X, y=y, confounds=confounds, groups=groups,
                        df=df)
