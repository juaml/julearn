# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from numpy.testing._private.utils import assert_array_almost_equal
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

from julearn.transformers.confounds import (ConfoundRemover,
                                            TargetConfoundRemover)

# TODO: change this values, the confounds are perect regressions
X = pd.DataFrame({
    'a': np.arange(10),
    'b': np.arange(10, 20),
    'c': np.arange(30, 40),
    'd': np.arange(40, 50),
    'e': np.arange(40, 50),
    'f': np.arange(40, 50),
})

y = np.arange(10)


def test_confound_removal_methods():
    """Test two different confound removal methods"""
    c_tests = [
        [X.copy(), ['e', 'f']],
        [X.drop(columns='d').copy(), ['f']]
    ]
    for _X, confounds in c_tests:
        n_confounds = len(confounds)
        features = _X.drop(columns=confounds).columns
        models = [LinearRegression(), RandomForestRegressor(n_estimators=5)]

        for model_to_remove in models:
            confound_remover = ConfoundRemover(model_confound=model_to_remove)

            np.random.seed(42)
            df_c_removed = confound_remover.fit_transform(
                _X, n_confounds=n_confounds)
            print(df_c_removed)
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

            assert_array_almost_equal(df_c_removed, df_removed_X.values)


def test_TargetConfoundRemover():
    confound_names = ['e', 'f']
    confounds = X[confound_names]
    n_confounds = len(confound_names)
    X_feat = X.drop(columns=confound_names)

    target_remover = TargetConfoundRemover()
    np.random.seed(42)
    y_transformed = target_remover.fit_transform(
        X_feat, y, n_confounds=n_confounds)
    np.random.seed(42)
    y_pred = (LinearRegression()
              .fit(confounds, y)
              .predict(confounds)
              )
    assert_array_almost_equal(y_transformed, y - y_pred)
