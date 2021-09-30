# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

from julearn.transformers.confounds import (BaseConfoundRemover,
                                            ConfoundRemover,
                                            TargetConfoundRemover)

# TODO: change this values, the confounds are perect regressions
X = pd.DataFrame({
    'a': np.arange(10),
    'b': np.array([0, 2, 1, 3, 3.5, 5.5, 6, 7, 7.6, 10]),
    'c': np.array([10, 11, 12, 14, 15, 16.5, 17.2, 18.3, 18.9, 20]),
})

y = np.array([20, 25, 22, 23.6, 24.9, 26, 27.5, 28, 28.8, 29.4])


def test_BaseConfoundRemover():
    bc = BaseConfoundRemover()
    bc.fit([1])
    bc.transform([1])
    bc.fit_transform([1])
    bc.will_drop_confounds()


def test_confound_removal_methods():
    """Test two different confound removal methods"""
    c_tests = [
        [X.drop(columns='c'), ['b']],
        [X, ['b', 'c']],
    ]
    for _X, confounds in c_tests:
        n_confounds = len(confounds)
        features = _X.drop(columns=confounds).columns
        models = [LinearRegression(),
                  RandomForestRegressor(n_estimators=5)
                  ]

        for model_to_remove in models:
            confound_remover = ConfoundRemover(model_confound=model_to_remove)

            np.random.seed(42)
            df_c_removed = (confound_remover
                            .fit(_X, n_confounds=n_confounds)
                            .transform(_X)
                            )
            np.random.seed(42)
            arr_c_removed = confound_remover.fit_transform(
                _X.values, n_confounds=n_confounds)
            np.random.seed(42)
            c_regressions = [
                clone(model_to_remove).fit(_X.loc[:, confounds],
                                           _X.loc[:, feature])
                for feature in features
            ]

            df_pred_X = _X.drop(columns=confounds).copy()

            # Test that each model inside of the confound removal
            # is the same as if we would have trained the same model
            # in sklearn

            for i_model, m_model, feature in zip(
                    confound_remover.models_confound_,
                    c_regressions,
                    features):

                manual_pred = m_model.predict(_X.loc[:, confounds])
                df_pred_X[feature] = manual_pred

                assert_array_equal(
                    i_model.predict(_X.loc[:, confounds]),
                    manual_pred
                )

            df_removed_X = (_X.drop(columns=confounds) - df_pred_X)

            assert_array_almost_equal(arr_c_removed, df_c_removed)
            assert_array_almost_equal(df_c_removed, df_removed_X.values)


def test_TargetConfoundRemover():
    confound_names = ['b', 'c']
    confounds = X[confound_names]
    n_confounds = len(confound_names)

    target_remover = TargetConfoundRemover()
    np.random.seed(42)
    y_transformed = target_remover.fit_transform(
        X, y, n_confounds=n_confounds)
    np.random.seed(42)
    y_pred = (LinearRegression()
              .fit(confounds, y)
              .predict(confounds)
              )
    assert_array_almost_equal(y_transformed, y - y_pred)


def test_thresholding():
    cr_no_thresh = ConfoundRemover()
    cr_thresh = ConfoundRemover(threshold=1e-2)
    _X = np.random.normal(size=[300, 2])
    _X = np.c_[_X, _X[:, 1]]
    _y = np.random.normal(size=300)
    res_no_thresh = cr_no_thresh.fit_transform(_X, _y, n_confounds=1)
    res_thresh = cr_thresh.fit_transform(_X, _y, n_confounds=1)

    assert res_no_thresh.min().min() < 1e-2
    assert res_thresh[res_thresh > 0].min() >= 1e-2


def test_apply_to():

    cr = ConfoundRemover()

    res_cr = cr.fit_transform(X, y, n_confounds=1, apply_to=slice(0, 1))
    assert_array_equal(res_cr[:, 1], X.iloc[:, 1])
    assert not ((res_cr == X.iloc[:, :-1]).all().all())


def test_n_confounds_nonewrong():

    cr = ConfoundRemover()

    with pytest.warns(
            RuntimeWarning,
            match='Number of confounds is 0 or below'):

        cr.fit(X, y, n_confounds=0)

    with pytest.warns(
            RuntimeWarning,
            match='Number of confounds is 0 or below'):

        cr.fit(X, y, n_confounds=-1)
