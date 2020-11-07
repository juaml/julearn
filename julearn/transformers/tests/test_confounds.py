# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from numpy.testing import assert_array_equal

from julearn.transformers.confounds import DataFrameConfoundRemover
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

X = pd.DataFrame({
    'a__:type:__continuous': np.arange(10),
    'b__:type:__continuous': np.arange(10, 20),
    'c__:type:__confound': np.arange(30, 40),
    'd__:type:__confound': np.arange(40, 50),
    'e__:type:__categorical': np.arange(40, 50),
    'f__:type:__categorical': np.arange(40, 50),
})

y = np.arange(10)


def test__apply_threshold():
    vals = pd.DataFrame([1e-4, 1e-2, 1e-1, 0, 1])
    confound_remover = DataFrameConfoundRemover(threshold=1e-2)
    out_pos_vals = confound_remover._apply_threshold(vals)
    out_neg_vals = confound_remover._apply_threshold(-vals)

    assert_frame_equal(
        out_pos_vals[[True, True, False, False, False]],
        out_neg_vals[[True, True, False, False, False]])

    assert (out_pos_vals.values == [0, 0, 1e-1, 0, 1]).all
    assert (out_neg_vals.values == [0, 0, -1e-1, 0, -1]).all


def test_confound_auto_find_conf():

    for _X, confounds in [
        [X.copy(), ['c__:type:__confound',
                    'd__:type:__confound']],
        [X.drop(columns='d__:type:__confound').copy(),
         ['c__:type:__confound']]
    ]:
        features = _X.drop(columns=confounds).columns

        for model_to_remove in [
                LinearRegression(), RandomForestRegressor(n_estimators=5)]:
            confound_remover = DataFrameConfoundRemover(
                model_confound=model_to_remove)

            np.random.seed(42)

            df_cofound_removed = confound_remover.fit_transform(_X)
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

            # After confound removal the confound should be removed
            assert (df_cofound_removed.columns == _X.drop(
                columns=confounds).columns).all()

            assert_frame_equal(df_cofound_removed, df_confound_removed_manual)


def test_confound_set_confounds():

    confounds_list = [
        'a__:type:__continuous',
        ['a__:type:__continuous'],
        ['a__:type:__continuous', 'b__:type:__continuous'],

    ]
    for model_to_remove in [
            LinearRegression(), RandomForestRegressor(n_estimators=5)]:
        for confounds in confounds_list:

            features = X.drop(columns=confounds).columns
            confound_remover = DataFrameConfoundRemover(
                model_confound=model_to_remove, confounds_match=confounds)

            np.random.seed(42)
            df_cofound_removed = confound_remover.fit_transform(X)

            np.random.seed(42)
            conf_as_feat = confounds if type(
                confounds) is list else [confounds]
            confound_regressions = [
                clone(model_to_remove).fit(X.loc[:, conf_as_feat],
                                           X.loc[:, feature])
                for feature in features
            ]
            df_confound_removed_manual = (X
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
                    X.loc[:, conf_as_feat])
                df_confound_removed_manual[feature] = manual_pred

                assert_array_equal(
                    internal_model.predict(X.loc[:, conf_as_feat]),
                    manual_pred
                )

            df_confound_removed_manual = (X.drop(columns=confounds)
                                          - df_confound_removed_manual)
            # After confound removal the confound should be removed
            assert (df_cofound_removed.columns == X.drop(
                columns=confounds).columns).all()

            assert_frame_equal(df_cofound_removed, df_confound_removed_manual)
