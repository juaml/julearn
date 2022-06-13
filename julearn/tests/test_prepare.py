# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from seaborn import load_dataset

from julearn.pipeline import _create_extended_pipeline

import pytest

from julearn.prepare import (prepare_input_data,
                             prepare_model_params,
                             _prepare_hyperparams)


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

    labeled_y = (y == pos_labels).astype(np.int64)
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

    labeled_y = np.isin(y, pos_labels).astype(np.int64)
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
    X = columns[:-2]
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
                       match=r"DataFrame columns must be strings"):
        X = 2
        y = columns[6]
        data = np.random.rand(4, 10)
        int_columns = [f'f_{x}' for x in range(data.shape[1] - 1)] + [0]

        X = columns[:-2]
        y = columns[-1]
        df_wrong_cols = pd.DataFrame(data=data, columns=int_columns)

        prepared = prepare_input_data(
            X=X, y=y, confounds=None, df=df_wrong_cols, pos_labels=None,
            groups=None)

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
    with pytest.warns(RuntimeWarning, match='contains the target'):
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


def test_prepare_model_params():
    preprocess_steps_features = [('zscore', StandardScaler()),
                                 ]
    model = ('svm', SVC())

    model_params = {'svm__kernel': 'linear'}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    pipeline = prepare_model_params(model_params, pipeline)
    assert pipeline['svm'].get_params()['kernel'] == 'linear'

    model_params = {
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svm__kernel': 'linear'}
    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    pipeline = prepare_model_params(model_params, pipeline)
    assert pipeline.cv.n_splits == 5  # sklearn cv default
    assert isinstance(pipeline, GridSearchCV)
    assert 'svm__C' in pipeline.param_grid
    assert 'svm__kernel' not in pipeline.param_grid

    model_params = {
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svm__gamma': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,
                       10, 100, 1000],
        'svm__kernel': 'rbf',
        'search': 'random',
        'search_params': {'n_iter': 50},
        'cv': 5
    }
    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    pipeline = prepare_model_params(model_params, pipeline)

    assert pipeline.cv.n_splits == 5
    assert isinstance(pipeline, RandomizedSearchCV)
    assert 'svm__C' in pipeline.param_distributions
    assert 'svm__gamma' in pipeline.param_distributions
    assert 'svm__kernel' not in \
        pipeline.param_distributions

    model_params = {'svm__kernel': 'linear', 'cv': 2}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    with pytest.warns(RuntimeWarning, match='search CV was specified'):
        pipeline = prepare_model_params(model_params, pipeline)

    model_params = {'svm__kernel': 'linear', 'scoring': 'accuracy'}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    with pytest.warns(RuntimeWarning, match='search scoring was specified'):
        pipeline = prepare_model_params(model_params, pipeline)

    model_params = {'svm__kernel': 'linear', 'search': 'grid'}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    with pytest.warns(RuntimeWarning, match='search method was specified'):
        pipeline = prepare_model_params(model_params, pipeline)

    model_params = {'svm__C': [0, 1], 'search': 'wrong'}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    with pytest.raises(ValueError, match='not a valid julearn searcher'):
        pipeline = prepare_model_params(model_params, pipeline)

    model_params = {'svm__C': [0, 1], 'search': GridSearchCV}

    pipeline = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_features,
        preprocess_transformer_target=None,
        preprocess_steps_confounds=None,
        model=model,
        confounds=None,
        categorical_features=None)
    with pytest.warns(RuntimeWarning,
                      match=f'{model_params["search"]} is not'
                      ' a registered searcher.'):
        pipeline = prepare_model_params(model_params, pipeline)


def test_pick_regexp():
    """Test picking columns by regexp"""
    data = np.random.rand(5, 10)
    columns = [
        '_a_b_c1_',
        '_a_b_c2_',
        '_a_b2_c3_',
        '_a_b2_c4_',
        '_a_b3_c5_',
        '_a_b3_c6_',
        '_a3_b2_c7_',
        '_a2_b_c7_',
        '_a2_b_c8_',
        '_a2_b_c9_'
    ]

    X = columns[: -1]
    y = columns[-1]
    confounds = None
    df = pd.DataFrame(data=data, columns=columns)

    prepared = prepare_input_data(X=X, y=y, confounds=confounds, df=df,
                                  pos_labels=None, groups=None)

    df_X_conf, df_y, _, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert y not in df_X_conf.columns
    assert df_y.name == y
    assert len(confound_names) == 0

    prepared = prepare_input_data(X=[':'], y=y, confounds=confounds, df=df,
                                  pos_labels=None, groups=None)

    df_X_conf, df_y, _, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert y not in df_X_conf.columns
    assert df_y.name == y
    assert len(confound_names) == 0

    X = columns[: 6]
    y = '_a3_b2_c7_'
    confounds = columns[-3:]
    prepared = prepare_input_data(X=[':'], y=y, confounds=confounds, df=df,
                                  pos_labels=None, groups=None)

    df_X_conf, df_y, _, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert all([x in df_X_conf.columns for x in confounds])
    assert y not in df_X_conf.columns
    assert df_y.name == y
    assert len(confound_names) == 3
    assert all([x in confound_names for x in confounds])

    X = columns[: 6]
    y = '_a3_b2_c7_'
    confounds = columns[-3: -1]
    groups = columns[-1]
    prepared = prepare_input_data(X=[':'], y=y, confounds=confounds, df=df,
                                  pos_labels=None, groups=groups)

    df_X_conf, df_y, df_groups, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert all([x in df_X_conf.columns for x in confounds])
    assert y not in df_X_conf.columns
    assert groups not in df_X_conf.columns
    assert df_y.name == y
    assert df_groups.name == groups
    assert len(confound_names) == 2
    assert all([x in confound_names for x in confounds])

    X = columns[: 6]
    y = '_a3_b2_c7_'
    confounds = columns[-3:]
    prepared = prepare_input_data(X=['_a_.*'], y=y, confounds='_a2_.*', df=df,
                                  pos_labels=None, groups=None)

    df_X_conf, df_y, _, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert all([x in df_X_conf.columns for x in confounds])
    assert y not in df_X_conf.columns
    assert df_y.name == y
    assert len(confound_names) == 3
    assert all([x in confound_names for x in confounds])

    X = columns[: 6]
    y = '_a3_b2_c7_'
    confounds = columns[-3:]
    prepared = prepare_input_data(X=['.*_b_.*', '.*a_b2_.*', '.*b3_.*'], y=y,
                                  confounds='_a2_.*', df=df,
                                  pos_labels=None, groups=None)

    df_X_conf, df_y, _, confound_names = prepared

    assert all([x in df_X_conf.columns for x in X])
    assert all([x in df_X_conf.columns for x in confounds])
    assert y not in df_X_conf.columns
    assert df_y.name == y
    assert len(confound_names) == 3
    assert all([x in confound_names for x in confounds])


def test__prepare_hyperparams():
    X = load_dataset('iris')
    y = X.pop('species')

    preprocess_steps_features = [('pca', PCA()),
                                 ]
    model = ('svm', SVC())

    grids = [{'svm__kernel': 'linear'},
             {'svm__kernel': ['linear']},
             {'svm__kernel': ['linear', 'rbf']},
             {'pca__n_components': [.2, .3]},
             {'pca__n_components': .2},
             {'pca__n_components': [.2]},
             {'svm__kernel': ['linear', 'rbf'],
              'pca__n_components': .2,
              }
             ]

    list_should_be_tuned = [False, False, True, True, False, False, True]
    for param_grid, should_be_tuned in zip(grids, list_should_be_tuned):
        pipeline = _create_extended_pipeline(
            preprocess_steps_features=preprocess_steps_features,
            preprocess_transformer_target=None,
            preprocess_steps_confounds=None,
            model=model,
            confounds=None,
            categorical_features=None)

        to_tune = _prepare_hyperparams(param_grid, pipeline)
        needs_tuning = len(to_tune) > 0
        if needs_tuning:
            pipeline = GridSearchCV(pipeline, param_grid=to_tune)

        with pytest.warns(None) as record:
            pipeline.fit(X, y)

        assert len(record) == 0
        assert needs_tuning == should_be_tuned

        if not needs_tuning:
            for param, val in param_grid.items():
                val = val[0] if type(val) == list else val
                assert pipeline.get_params()[param] == val
