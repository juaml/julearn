# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from julearn.transformers.confounds import DataFrameConfoundRemover
from sklearn.linear_model import LinearRegression

X = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                  'cheese__:type:__confound': np.arange(50, 60)})

X_multi = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                        'cookie__:type:__confound': np.arange(40, 50),
                        'cheese__:type:__confound': np.arange(50, 60)})

X_with_types = pd.DataFrame({
    'a__:type:__continuous': np.arange(10),
    'b__:type:__continuous': np.arange(10, 20),
    'c__:type:__confound': np.arange(30, 40),
    'd__:type:__confound': np.arange(40, 50),
    'e__:type:__categorical': np.arange(40, 50),
    'f__:type:__categorical': np.arange(40, 50),
})

y = np.arange(10)


def test_DataFrameConfoundRemover_suffix_one_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_trans = remover.fit_transform(X)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_DataFrameConfoundRemover_suffix_multi_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_trans = remover.fit_transform(X_multi)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_confound_auto_find_conf():
    confounds = ['c__:type:__confound',
                 'd__:type:__confound']
    features = X_with_types.drop(columns=confounds).columns
    confound_remover = DataFrameConfoundRemover()

    np.random.seed(42)
    confound_remover.fit_transform(X_with_types)

    np.random.seed(42)
    confound_regressions = [
        LinearRegression().fit(X_with_types.loc[:, confounds],
                               X_with_types.loc[:, feature])
        for feature in features
    ]
    for internal_model, confound_regression in zip(
            confound_remover.models_confound_, confound_regressions):
        assert_array_equal(
            internal_model.predict(X_with_types.loc[:, confounds]),
            confound_regression.predict(X_with_types.loc[:, confounds])
        )


def test_confound_set_confounds():

    confounds_list = [
        'a__:type:__continuous',
        ['a__:type:__continuous', 'b__:type:__continuous'],

    ]
    for confounds in confounds_list:

        features = X_with_types.drop(columns=confounds).columns
        confound_remover = DataFrameConfoundRemover(confounds=confounds)

        np.random.seed(42)
        confound_remover.fit_transform(X_with_types)

        np.random.seed(42)
        conf_as_feat = confounds if type(confounds) is list else [confounds]
        confound_regressions = [
            LinearRegression().fit(X_with_types.loc[:, conf_as_feat],
                                   X_with_types.loc[:, feature])
            for feature in features
        ]
        for internal_model, confound_regression in zip(
                confound_remover.models_confound_, confound_regressions):
            assert_array_equal(
                internal_model.predict(X_with_types.loc[:, conf_as_feat]),
                confound_regression.predict(X_with_types.loc[:, conf_as_feat])
            )
