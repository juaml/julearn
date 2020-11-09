# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression

from .. utils import raise_error, pick_columns


class DataFrameConfoundRemover(TransformerMixin, BaseEstimator):
    def __init__(
        self, model_confound=None, confounds_match='.*__:type:__confound',
        threshold=None
    ):
        """Transformer which can use pd.DataFrames and remove the confounds
        from the features by subtracting the predicted features
        given the confounds from the actual features.

        Parameters
        ----------
        model_confound : sklearn.base.BaseEstimator
            Model used to predict all features independentley
            using the confounds as features. The predictions of these models
            are then subtracted from each feature,
            default LinearRegression()
        confounds_match : list[str] or str
            A string representing a regular expression by which the confounds
            can be detected from the column names.
            You can use the exact column names or another regex.
            The default follows the naming convention inside of julearn,
            default '.*__:type:__*.'
        threshold : float or None
            All residual values after confound removal which fall under
            the threshold will be set to 0.
            None means that no threshold will be applied,
            default None
        """
        if model_confound is None:
            model_confound = LinearRegression()
        self.model_confound = model_confound
        self.confounds_match = confounds_match
        self.threshold = threshold

    def fit(self, X, y=None):
        df_X, ser_confound = self._split_into_X_confound(X)
        self.support_mask_ = pd.Series(False, index=X.columns,
                                       dtype=bool)
        self.support_mask_[df_X.columns] = True
        self.support_mask_ = self.support_mask_.values

        def fit_confound_models(X):
            _model = clone(self.model_confound)
            _model.fit(ser_confound.values, X)
            return _model

        self.models_confound_ = df_X.apply(fit_confound_models, axis=0,
                                           result_type='reduce')
        return self

    def transform(self, X):
        df_X, df_confounds = self._split_into_X_confound(X)
        df_X_prediction = pd.DataFrame(
            [model.predict(df_confounds)
             for model in self.models_confound_.values],
            index=df_X.columns,
            columns=df_X.index,
        ).T
        residuals = df_X - df_X_prediction
        return self._apply_threshold(residuals)

    def get_support(self, indices=False):
        if indices:
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_

    def _split_into_X_confound(self, X):
        """splits the original X input into the reals features (X) and confound
        """
        if not isinstance(X, pd.DataFrame):
            raise_error(
                'DataFrameConfoundRemover only supports DataFrames as X')

        df_X = X.copy()

        self.detected_confounds_ = pick_columns(
            self.confounds_match, df_X.columns)
        if self.detected_confounds_ == []:
            raise_error(f'no confound was found using the suffix'
                        f'{self.suffix} in   the columns {X.columns}')
        df_confounds = df_X.loc[:, self.detected_confounds_]
        df_X = df_X.drop(columns=self.detected_confounds_)
        return df_X, df_confounds

    def _apply_threshold(self, residuals):
        """Rounds residuals to 0 when smaller than
        the previously described absolute threshold.
        This is done to prevent correlated rounding errors
        """
        if self.threshold is not None:
            # Accounting for correlated rounding errors for very small
            # residuals
            residuals = residuals.applymap(
                lambda x: 0 if abs(x) <= self.threshold else x
            )
        return residuals
