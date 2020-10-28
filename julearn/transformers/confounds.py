import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression


class DataFrameConfoundRemover(TransformerMixin, BaseEstimator):
    def __init__(
        self, model_confound=None, confounds='use_suffix',
        threshold=None, suffix='__:type:__confound'
    ):
        """Model to regress out/remove confound
        by a given model from the features.

        Parameters
        ----------
        model_confound : [sklearn model]
                         Model by which we predict X
                         given the confound to later compute the residuals.
        confounds   : [int, str]
                         The indx in ndarray or
                         the column name in a df of the confound.

        threshold      : [float]
                         Threshold under which all values are rounded to 0.

        """
        if model_confound is None:
            model_confound = LinearRegression()
        self.model_confound = model_confound
        self.confounds = confounds
        self.threshold = threshold
        self.suffix = suffix

        if type(confounds) != list:
            self.multiple_confounds = False
        elif len(confounds) > 1:
            self.multiple_confounds = True
        else:
            self.multiple_confounds = False

    def fit(self, X, y=None):
        """Modelling X given confound in the internal model.

        Parameters

        """

        df_X, ser_confound = self.split_into_X_confound(X)
        if type(X) == np.ndarray:
            idx_kept = list(df_X.columns)
            self.support_mask_ = np.zeros(len(X[0]), dtype=bool)
            self.support_mask_[idx_kept] = True

        else:
            self.support_mask_ = pd.Series(False, index=X.columns,
                                           dtype=bool)
            self.support_mask_[df_X.columns] = True
            self.support_mask_ = self.support_mask_.values

        def fit_confound_models(X):
            _model = clone(self.model_confound)
            if self.multiple_confounds:
                _model.fit(ser_confound.values, X)
            else:
                _model.fit(ser_confound.values.reshape(-1, 1), X)
            return _model

        self.models_confound_ = df_X.apply(fit_confound_models, axis=0)
        return self

    def transform(self, X):
        """Cleaning/Subtracting the predicted X given the confound out of X.


        """
        df_X, ser_confound = self.split_into_X_confound(X)

        conf_as_x = (
            ser_confound.values
            if self.multiple_confounds
            else ser_confound.values.reshape(-1, 1)
        )
        df_X_prediction = pd.DataFrame(
            [model.predict(conf_as_x)
             for model in self.models_confound_.values],
            index=df_X.columns,
            columns=df_X.index,
        ).T
        residuals = df_X - df_X_prediction
        return self.apply_threshold(residuals)

    def split_into_X_confound(self, X):
        """splits the original X input into the reals features (X) and confound
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                'DataFrameConfoundRemover only supports DataFrames as X')

        df_X = X.copy()
        ser_confound = None
        if self.confounds == 'use_suffix':

            self.detected_confounds_ = [column
                                        for column in df_X.columns
                                        if column.endswith(self.suffix)
                                        ]
            if self.detected_confounds_ == []:
                raise ValueError(f'no confound was found using the suffix'
                                 f'{self.suffix} in   the columns {X.columns}')
            if len(self.detected_confounds_) == 1:
                self.multiple_confounds = False
                self.detected_confounds_ = self.detected_confounds_[0]
                ser_confound = df_X.pop(self.detected_confounds_)

            else:
                self.multiple_confounds = True
                ser_confound = df_X.loc[:, self.detected_confounds_]
                df_X = df_X.drop(columns=self.detected_confounds_)

        else:

            if type(self.confounds) == str:
                ser_confound = df_X.pop(self.confounds)
            elif type(self.confounds) == int:
                ser_confound = df_X.iloc[:, self.confounds]
                df_X = df_X.drop(columns=self.confounds)
            elif type(self.confounds) == list:
                ser_confound = df_X.loc[:, self.confounds]
                df_X = df_X.drop(columns=self.confounds)

        return df_X, ser_confound

    def apply_threshold(self, residuals):
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

    def get_support(self, indices=False):
        if indices:
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_
