# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from julearn.transformers.confounds import (ConfoundRemover,
                                            TargetConfoundRemover)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

X = pd.DataFrame({
    'a': np.arange(10),
    'b': np.arange(10, 20),
    'c': np.arange(30, 40),
    'd': np.arange(40, 50),
    'e': np.arange(40, 50),
    'f': np.arange(40, 50),
})

y = np.arange(10)


def test__apply_threshold():
    """Test apply thershold on residuals"""
    vals = np.array([1e-4, 1e-2, 1e-1, 0, 1])
    confound_remover = ConfoundRemover(threshold=1e-2)
    out_pos_vals = confound_remover._apply_threshold(vals)
    out_neg_vals = confound_remover._apply_threshold(-vals)

    assert_array_equal(
        out_pos_vals[[True, True, False, False, False]],
        out_neg_vals[[True, True, False, False, False]])

    assert_array_equal(out_pos_vals, [0, 0, 1e-1, 0, 1])
    assert_array_equal(out_neg_vals, [0, 0, -1e-1, 0, -1])


def test_confound_removal_methods():
    """Test two different confound removal methods"""
    for _X, confounds in [
        [X.copy(), ['c', 'd']],
        [X.drop(columns='d').copy(), ['c']]
    ]:
        features = _X.drop(columns=confounds).columns

        for model_to_remove in [
                LinearRegression(), RandomForestRegressor(n_estimators=5)]:

            confound_remover = ConfoundRemover(model_confound=model_to_remove)

            np.random.seed(42)
            df_confound_removed = confound_remover.fit_transform(
                _X.drop(columns=confounds), confounds=_X[confounds])

            np.random.seed(42)
            confound_regressions = [
                clone(model_to_remove).fit(_X.loc[:, confounds],
                                           _X.loc[:, feature])
                for feature in features
            ]

            df_confound_removed_manual = (_X
                                          .drop(columns=confounds)
                                          .copy()
                                          )
            # Test that each model inside of the confound removal
            # is the same as if we would have trained the same model
            # in sklearn
            for internal_model, confound_regression, feature in zip(
                    confound_remover.models_confound_,
                    confound_regressions,
                    features):

                manual_pred = confound_regression.predict(
                    _X.loc[:, confounds])
                df_confound_removed_manual[feature] = manual_pred

                assert_array_equal(
                    internal_model.predict(_X.loc[:, confounds]),
                    manual_pred
                )
            df_confound_removed_manual = (_X.drop(columns=confounds)
                                          - df_confound_removed_manual)

            assert_array_equal(
                df_confound_removed, df_confound_removed_manual.values)


def test_TargetConfoundRemover():
    confound_names = ['c', 'd']
    confounds = X[confound_names]
    X_feat = X.drop(columns=confound_names)

    target_remover = TargetConfoundRemover()
    np.random.seed(42)
    y_transformed = target_remover.fit_transform(
        X_feat, y, confounds=confounds)
    np.random.seed(42)
    y_pred = (LinearRegression()
              .fit(confounds, y)
              .predict(confounds)
              )
    assert_array_equal(y_transformed, y - y_pred)
